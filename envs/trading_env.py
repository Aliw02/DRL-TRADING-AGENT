import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from utils.logger import get_logger
from config.init import config

logger = get_logger(__name__)

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, initial_balance=None, sequence_length=None):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1

        self.initial_balance = initial_balance or config.get('environment.initial_balance', 10000)
        self.sequence_length = sequence_length or config.get('environment.sequence_length', 150)
        self.commission_pct = config.get('environment.commission_pct', 0.0005)
        
        self.action_penalty = config.get('environment.action_penalty', 0.1)
        self.profit_bonus_multiplier = config.get('environment.profit_bonus', 2.0)

        self.action_space = spaces.Discrete(3)
        self.feature_cols = [col for col in self.df.columns if col not in ['open', 'high', 'low', 'close', 'time']]
        obs_dim = len(self.feature_cols) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.sequence_length, obs_dim), dtype=np.float32)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.sequence_length
        self.position = 0
        self.entry_price = 0
        self.portfolio_value = self.initial_balance
        self.trade_count = 0
        
        return self._next_observation(), {}
    
    def step(self, action):
        prev_portfolio_value = self.portfolio_value
        realized_pnl = self._take_action(action)
        self._update_portfolio_value()
        reward = self._get_reward(prev_portfolio_value, realized_pnl)
        
        self.current_step += 1
        
        done = (self.portfolio_value < self.initial_balance * 0.5) or (self.current_step >= self.max_steps)

        obs = self._next_observation()
        info = {'equity': self.portfolio_value, 'trade_count': self.trade_count}
        
        return obs, reward, done, False, info

    def _update_portfolio_value(self):
        if self.position == 1:
            current_price = self.df.loc[self.current_step, 'close']
            unrealized_pnl = (current_price - self.entry_price)
            self.portfolio_value = self.trade_start_balance + unrealized_pnl

    def _get_reward(self, prev_portfolio_value, realized_pnl):
        holding_reward = self.portfolio_value - prev_portfolio_value
        profit_bonus = 0
        if realized_pnl > 0:
            profit_bonus = realized_pnl * self.profit_bonus_multiplier
        reward = holding_reward + profit_bonus + realized_pnl
        return reward

    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, 'close']
        realized_pnl = 0
        
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
            self.trade_start_balance = self.portfolio_value
            self.trade_count += 1
            self.portfolio_value -= self.action_penalty

        elif action == 2 and self.position == 1:
            exit_price = current_price
            realized_pnl = (exit_price - self.entry_price)
            
            commission = self.trade_start_balance * self.commission_pct
            
            self.portfolio_value += realized_pnl - commission
            
            self.position = 0
            self.entry_price = 0
            
        return realized_pnl
            
    def _next_observation(self):
        # --- THE CRITICAL FIX IS HERE ---
        # The slice should end at 'self.current_step' (inclusive), not 'self.current_step + 1'
        # This guarantees the length is always exactly 'self.sequence_length'.
        start_idx = self.current_step - self.sequence_length + 1
        end_idx = self.current_step
        
        # Ensure start index is not negative
        if start_idx < 0:
            start_idx = 0
            
        features_df = self.df.loc[start_idx:end_idx, self.feature_cols]
        
        position_feature = np.zeros((len(features_df), 1))
        if self.position == 1:
            position_feature.fill(1)
            
        obs = np.concatenate([features_df.values, position_feature], axis=1)

        if len(obs) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(obs), obs.shape[1]))
            obs = np.vstack([padding, obs])
            
        return obs.astype(np.float32)

