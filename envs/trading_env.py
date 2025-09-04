# envs/trading_env.py (OPTIMIZED FOR CPU-BASED DATA HANDLING)
import gymnasium as gym
from gymnasium import spaces
import numpy as np # <-- We will use the standard numpy
import pandas as pd # <-- We will use the standard pandas
from utils.logger import get_logger
from config.init import config
from utils.metrics import calculate_sortino_ratio

logger = get_logger(__name__)

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df: pd.DataFrame, initial_balance=None, sequence_length=None): # Type hint for clarity
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1
        
        self.initial_balance = initial_balance or config.get('environment.initial_balance', 10000)
        self.sequence_length = sequence_length or config.get('environment.sequence_length', 150)
        self.commission_pct = config.get('environment.commission_pct', 0.0005)
        self.turnover_penalty = config.get('environment.turnover_penalty', 0.001)

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        self.feature_cols = [col for col in self.df.columns if col not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
        obs_dim = len(self.feature_cols) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.sequence_length, obs_dim), dtype=np.float32)
        
        self.reset()
    
    def _get_price(self, step):
        # Direct access using pandas is fast on CPU
        return self.df.loc[step, 'close']

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.sequence_length
        self.portfolio_value = self.initial_balance
        self.cash = self.initial_balance
        self.position_size = 0.0
        self.units_held = 0.0
        self.portfolio_return_history = [0.0] * self.sequence_length
        self.max_portfolio_value = self.initial_balance
        
        # No more conversions needed, obs is already a numpy array
        obs = self._next_observation()
            
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
        
        # No more conversions needed here either
        return obs, reward, done, False, info

    def _update_portfolio_value(self):
        current_price = self._get_price(self.current_step)
        position_market_value = self.units_held * current_price
        self.portfolio_value = self.cash + position_market_value
        
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value

    def _get_reward(self, turnover):
        returns_np_array = np.array(self.portfolio_return_history) # Already on CPU
        sortino_reward = calculate_sortino_ratio(returns_np_array)
        step_return = self.portfolio_return_history[-1]
        pnl_reward = np.tanh(step_return * 10)
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        drawdown_penalty = - (current_drawdown ** 2)
        turnover_penalty = self.turnover_penalty * turnover
        reward = ((sortino_reward * 0.5) + (pnl_reward * 0.1) + (drawdown_penalty * 0.4) - turnover_penalty)
        return 0.0 if np.isnan(reward) else float(reward)

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
        
        # .values on a pandas DataFrame returns a numpy array directly
        features_array = self.df.loc[start_idx:end_idx, self.feature_cols].values
        
        position_feature = np.full((len(features_array), 1), self.position_size, dtype=np.float32)
        obs = np.concatenate([features_array, position_feature], axis=1)
        return obs.astype(np.float32)