# envs/trading_env.py (ADVANCED & PRACTICAL VERSION)

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

        # --- Core Parameters ---
        self.initial_balance = initial_balance or config.get('environment.initial_balance', 10000)
        self.sequence_length = sequence_length or config.get('environment.sequence_length', 40)
        self.commission_pct = config.get('environment.commission_pct', 0.0005)
        self.slippage_pct = config.get('environment.slippage_pct', 0.0002)

        # --- State Variables ---
        self.action_space = spaces.Discrete(3)
        feature_df = self.df.drop(columns=['close'], errors='ignore')
        obs_dim = feature_df.shape[1] + 1 # features + position
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.sequence_length, obs_dim), dtype=np.float32)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.sequence_length
        self.position = 0
        self.entry_price = 0
        self.equity = self.initial_balance
        self.trade_count = 0
        
        # For drawdown calculation
        self.max_equity = self.initial_balance
        # For Sortino Ratio calculation
        self.equity_history = [self.initial_balance] * 50 # Keep track of last 50 equity values for returns calc
        
        return self._next_observation(), {}
    
    def step(self, action):
        # Store equity before taking the step for reward calculation
        prev_equity = self.equity
        
        # Update current equity based on market movement
        self._update_equity()
        
        # Execute the trade action
        self._take_action(action)
        
        # Update equity history and max equity for metrics
        self.equity_history.append(self.equity)
        self.equity_history.pop(0)
        self.max_equity = max(self.max_equity, self.equity)

        # Calculate the multi-objective reward
        reward = self._get_reward(prev_equity)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps or self.equity <= self.initial_balance * 0.5 # Stop if 50% loss

        obs = self._next_observation()
        info = {'equity': self.equity, 'trade_count': self.trade_count}
        
        return obs, reward, done, False, info

    def _update_equity(self):
        """Update equity based on the current price if a position is open."""
        if self.position == 1: # Long position
            current_price = self.df.loc[self.current_step, 'close']
            self.equity = self.initial_balance + (current_price - self.entry_price)

    def _get_reward(self, prev_equity):
        """
        A practical, multi-objective reward function.
        """
        # 1. PnL Reward: The most direct signal
        pnl_reward = (self.equity - prev_equity) / prev_equity

        # 2. Drawdown Penalty: Penalize for losing capital from the peak
        current_drawdown = (self.max_equity - self.equity) / self.max_equity
        # The penalty increases exponentially as drawdown gets worse
        drawdown_penalty = - (current_drawdown ** 2)
        
        # 3. Sortino Ratio Reward: Encourage good risk-adjusted returns
        returns = pd.Series(self.equity_history).pct_change().dropna()
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) > 1 and negative_returns.std() != 0:
            sortino_ratio = (returns.mean() / negative_returns.std()) * 0.1 # Scaled down
        else:
            sortino_ratio = 0
            
        # Combine the objectives with weights
        # The weights determine the priorities. Here, we prioritize PnL and capital preservation.
        reward = (pnl_reward * 100) + (drawdown_penalty * 2.0) + (sortino_ratio)
        
        return reward if np.isfinite(reward) else 0.0

    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, 'close']
        
        if action == 1 and self.position == 0: # Buy
            self.position = 1
            self.entry_price = current_price * (1 + self.slippage_pct)
            self.initial_balance -= self.initial_balance * self.commission_pct
            self.trade_count += 1

        elif action == 2 and self.position == 1: # Sell (close long)
            profit = (current_price - self.entry_price)
            self.initial_balance += profit
            self.initial_balance -= self.initial_balance * self.commission_pct
            self.position = 0
            self.entry_price = 0
            # On closing a trade, reset the max_equity to the new balance to "bank" the profits
            self.max_equity = self.initial_balance 
            self.equity = self.initial_balance
            self.trade_count += 1
            
    def _next_observation(self):
        start_idx = max(0, self.current_step - self.sequence_length + 1)
        end_idx = self.current_step
        
        features_df = self.df.loc[start_idx:end_idx].drop(columns=['close'], errors='ignore')
        
        position_feature = np.zeros((len(features_df), 1))
        if self.position == 1:
            position_feature.fill(1)
            
        obs = np.concatenate([features_df.values, position_feature], axis=1)

        if len(obs) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(obs), obs.shape[1]))
            obs = np.vstack([padding, obs])
            
        return obs.astype(np.float32)