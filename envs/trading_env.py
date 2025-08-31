# envs/trading_env.py (FINAL CORRECTED VERSION WITH CUPY FIX)
import gymnasium as gym
from gymnasium import spaces
from utils.accelerator import np, IS_GPU_AVAILABLE
from utils.logger import get_logger
from config.init import config
from utils.metrics import calculate_sharpe_ratio
import numpy as np_cpu  # Import regular numpy for explicit conversions

logger = get_logger(__name__)

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, initial_balance=None, sequence_length=None):
        super(TradingEnv, self).__init__()
        
        # We reset the index here to ensure it's a simple integer index (0, 1, 2, ...)
        # This makes .iloc more reliable and avoids potential issues with DatetimeIndex.
        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1
        
        self.initial_balance = initial_balance or config.get('environment.initial_balance', 10000)
        self.sequence_length = sequence_length or config.get('environment.sequence_length', 150)
        self.commission_pct = config.get('environment.commission_pct', 0.0005)
        self.turnover_penalty = config.get('environment.turnover_penalty', 0.001)

        # Use regular numpy for action and observation spaces to ensure compatibility
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np_cpu.float32)
        
        self.feature_cols = [col for col in self.df.columns if col not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
        obs_dim = len(self.feature_cols) + 1
        self.observation_space = spaces.Box(low=-np_cpu.inf, high=np_cpu.inf, shape=(self.sequence_length, obs_dim), dtype=np_cpu.float32)
        
        self.reset()
    
    def _get_price(self, step):
        # Since we reset the index, we can reliably use .iloc
        if IS_GPU_AVAILABLE:
            return self.df['close'].iloc[step].item()
        else:
            return self.df.iloc[step]['close']

    def _ensure_numpy_array(self, array):
        """Convert CuPy array to NumPy array if needed."""
        if IS_GPU_AVAILABLE and hasattr(array, 'get'):
            # This is a CuPy array, convert it to NumPy
            return array.get()
        elif IS_GPU_AVAILABLE and hasattr(np, 'asnumpy'):
            # Alternative CuPy conversion method
            return np.asnumpy(array)
        else:
            # Already a NumPy array or using CPU
            return np_cpu.asarray(array)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.sequence_length
        self.portfolio_value = self.initial_balance
        self.cash = self.initial_balance
        self.position_size = 0.0
        self.units_held = 0.0
        self.portfolio_return_history = [0.0] * self.sequence_length 
        
        # Ensure the observation is returned as a NumPy array
        obs = self._next_observation()
        obs = self._ensure_numpy_array(obs)
        
        return obs, {}
    
    def step(self, action):
        target_position_size = float(action[0])
        prev_portfolio_value = self.portfolio_value
        turnover = abs(target_position_size - self.position_size)
        
        self._take_action(target_position_size)
        self._update_portfolio_value()
        
        step_return = (self.portfolio_value / prev_portfolio_value) - 1 if prev_portfolio_value != 0 else 0
        self.portfolio_return_history.append(step_return)
        self.portfolio_return_history.pop(0)

        reward = self._get_reward(turnover)
        self.current_step += 1
        done = (self.portfolio_value < self.initial_balance * 0.5) or (self.current_step >= self.max_steps)

        obs = self._next_observation()
        info = {'equity': self.portfolio_value, 'position_size': self.position_size}
        
        # Always ensure observation is a NumPy array for Stable-Baselines3 compatibility
        obs = self._ensure_numpy_array(obs)
            
        return obs, reward, done, False, info

    def _update_portfolio_value(self):
        current_price = self._get_price(self.current_step)
        position_market_value = self.units_held * current_price
        self.portfolio_value = self.cash + position_market_value

    def _get_reward(self, turnover):
        returns_array = np_cpu.array(self.portfolio_return_history)  # Use CPU numpy for metrics
        sharpe_reward = calculate_sharpe_ratio(returns_array)
        penalty = self.turnover_penalty * turnover
        return sharpe_reward - penalty

    def _take_action(self, target_position_size):
        current_price = self._get_price(self.current_step)
        if current_price == 0: return

        target_units = (self.portfolio_value * target_position_size) / current_price
        units_to_trade = target_units - self.units_held
        
        if units_to_trade != 0:
            cost_of_trade = abs(units_to_trade * current_price)
            commission = cost_of_trade * self.commission_pct
            self.cash -= (units_to_trade * current_price) + commission
            self.units_held += units_to_trade
            self.position_size = (self.units_held * current_price) / self.portfolio_value if self.portfolio_value != 0 else 0
            
    def _next_observation(self):
        start_idx = self.current_step - self.sequence_length + 1
        end_idx = self.current_step
        
        # Use .iloc for integer-location based slicing
        obs_slice = self.df.iloc[start_idx : end_idx + 1]
        features_array = obs_slice[self.feature_cols].values
        
        # Create position feature - ensure it's compatible with the features array type
        if IS_GPU_AVAILABLE:
            position_feature = np.full((len(features_array), 1), self.position_size, dtype=np.float32)
            obs = np.concatenate([features_array, position_feature], axis=1)
        else:
            position_feature = np_cpu.full((len(features_array), 1), self.position_size, dtype=np_cpu.float32)
            obs = np_cpu.concatenate([features_array, position_feature], axis=1)
        
        return obs.astype(np_cpu.float32) if not IS_GPU_AVAILABLE else obs.astype(np.float32)
