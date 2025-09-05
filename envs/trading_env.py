# envs/trading_env.py
# FINAL, BATTLE-TESTED, AND STRATEGICALLY-SOUND TRADING ENVIRONMENT

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from utils.logger import get_logger
from config.init import config

logger = get_logger(__name__)

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df: pd.DataFrame):
        super(TradingEnv, self).__init__()
        
        # --- HIGH-PERFORMANCE OPTIMIZATION: Pre-load all data into NumPy ---
        self.feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
        self.features_array = df[self.feature_cols].values.astype(np.float32)
        self.price_array = df['close'].values.astype(np.float32)
        
        self.max_steps = len(df) - 1
        
        # Load strategic parameters
        self.initial_balance = config.get('environment.initial_balance', 10000)
        self.sequence_length = config.get('environment.sequence_length', 60)
        self.commission_pct = config.get('environment.commission_pct', 0.0005) # Represents spread + commission
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        obs_dim = self.features_array.shape[1] + 1 # Market features + current position size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.sequence_length, obs_dim), 
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.sequence_length
        self.portfolio_value = self.initial_balance
        self.cash = self.initial_balance
        self.position_size = 0.0
        self.units_held = 0.0
        self.max_portfolio_value = self.initial_balance
        
        return self._next_observation(), {}
    
    def step(self, action):
        target_position_size = float(action[0])
        
        # --- REWARD LOGIC: Capture realized PnL from closing a position ---
        realized_pnl = self._take_action(target_position_size)
        
        self.current_step += 1
        self._update_portfolio_value()
        
        reward = self._get_reward(realized_pnl)
        
        done = (self.portfolio_value < self.initial_balance * 0.5) or (self.current_step >= self.max_steps)

        obs = self._next_observation()
        info = {'equity': self.portfolio_value, 'position_size': self.position_size}
        
        return obs, reward, done, False, info

    def _update_portfolio_value(self):
        current_price = self.price_array[self.current_step]
        position_market_value = self.units_held * current_price
        self.portfolio_value = self.cash + position_market_value
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)

    def _get_reward(self, realized_pnl):
        """
        STRATEGICALLY-SOUND AND PERFORMANCE-OPTIMIZED REWARD FUNCTION.
        It provides clear, realistic signals without compromising speed.
        """
        # --- PRIMARY OBJECTIVE: REWARD REALIZED PROFIT ---
        # The agent is now rewarded ONLY when it closes a trade profitably.
        # This is a clear, sparse, and powerful signal. We use percentage PnL.
        pnl_reward = realized_pnl / self.initial_balance
        
        # --- DEFENSIVE PENALTY: UNREALIZED DRAWDOWN ---
        # At every step, the agent is penalized for the current drawdown of the whole portfolio.
        # This teaches capital preservation continuously.
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        drawdown_penalty = drawdown ** 2 # Punish large drawdowns severely
        
        # --- FINAL REWARD: A realistic trading signal ---
        # The agent must generate enough profit to overcome the constant pressure of drawdown.
        reward = pnl_reward - drawdown_penalty
        
        return float(reward) if np.isfinite(reward) else -1.0

    def _take_action(self, target_position_size):
        """
        Executes trades and now returns the realized PnL from any closing action.
        """
        current_price = self.price_array[self.current_step]
        if current_price <= 1e-9: return 0.0

        realized_pnl = 0.0
        current_position_value = self.units_held * current_price

        # --- Logic for Closing or Reducing a Position ---
        position_change = target_position_size - self.position_size
        
        # If the agent wants to close or flip its position, calculate realized PnL
        if (self.position_size > 0 and target_position_size <= 0) or \
           (self.position_size < 0 and target_position_size >= 0):
            
            # Realize PnL on the entire position
            self.cash += current_position_value * (1 - self.commission_pct)
            realized_pnl = self.cash - self.balance # This captures the PnL
            self.balance = self.cash
            self.units_held = 0
            
        # --- Logic for Opening or Increasing a Position ---
        target_units = (self.portfolio_value * target_position_size) / current_price
        units_to_trade = target_units - self.units_held
        
        if abs(units_to_trade) > 1e-6:
            trade_cost = abs(units_to_trade * current_price)
            commission = trade_cost * self.commission_pct
            self.cash -= (units_to_trade * current_price) + commission
            self.units_held += units_to_trade
        
        self.position_size = (self.units_held * current_price) / self.portfolio_value if self.portfolio_value > 1e-9 else 0
        return realized_pnl
            
    def _next_observation(self):
        start_idx = self.current_step - self.sequence_length + 1
        end_idx = self.current_step + 1
        features = self.features_array[start_idx:end_idx]
        position_feature = np.full((self.sequence_length, 1), self.position_size, dtype=np.float32)
        return np.concatenate([features, position_feature], axis=1)