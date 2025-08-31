# envs/trading_env.py (FINAL PROFESSIONAL VERSION)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from utils.logger import get_logger
from config.init import config
from utils.metrics import calculate_sharpe_ratio

logger = get_logger(__name__)

class TradingEnv(gym.Env):
    """
    A professional trading environment for DRL agents that uses:
    - A dynamic observation space to accommodate any number of features.
    - A Sharpe Ratio-based reward function to optimize for risk-adjusted returns.
    - Commission and action penalties to simulate real-world trading costs.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, initial_balance=None, sequence_length=None):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1

        # Load parameters from the global config
        self.initial_balance = initial_balance or config.get('environment.initial_balance', 10000)
        self.sequence_length = sequence_length or config.get('environment.sequence_length', 150)
        self.commission_pct = config.get('environment.commission_pct', 0.0005)
        self.action_penalty = config.get('environment.action_penalty', 0.001)
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # --- DYNAMIC OBSERVATION SPACE ---
        # Automatically detects feature columns, including the new 'regime_prob_X' columns
        self.feature_cols = [col for col in self.df.columns if col not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
        obs_dim = len(self.feature_cols) + 1 # +1 for the current position
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.sequence_length, obs_dim), dtype=np.float32)
        
        logger.info(f"Environment initialized with observation space dimension: {obs_dim}")
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.sequence_length
        self.position = 0 # 0: no position, 1: long
        self.entry_price = 0
        self.portfolio_value = self.initial_balance
        self.trade_start_balance = self.initial_balance
        self.trade_count = 0
        # History of portfolio returns for Sharpe Ratio calculation
        self.portfolio_return_history = [0.0] * self.sequence_length 
        
        return self._next_observation(), {}
    
    def step(self, action):
        prev_portfolio_value = self.portfolio_value
        
        self._take_action(action)
        self._update_portfolio_value()
        
        # Calculate the return for the current step and update history
        step_return = (self.portfolio_value / prev_portfolio_value) - 1 if prev_portfolio_value != 0 else 0
        self.portfolio_return_history.append(step_return)
        self.portfolio_return_history.pop(0)

        # Calculate the professional, risk-adjusted reward
        reward = self._get_reward(action)
        
        self.current_step += 1
        
        # Check for termination conditions
        done = (self.portfolio_value < self.initial_balance * 0.5) or (self.current_step >= self.max_steps)

        obs = self._next_observation()
        info = {'equity': self.portfolio_value, 'trade_count': self.trade_count}
        
        return obs, reward, done, False, info

    def _update_portfolio_value(self):
        # If a position is open, update its value based on the current price
        if self.position == 1:
            current_price = self.df.loc[self.current_step, 'close']
            unrealized_pnl = (current_price - self.entry_price)
            self.portfolio_value = self.trade_start_balance + unrealized_pnl

    def _get_reward(self, action):
        # The primary reward is the Sharpe Ratio of recent performance
        sharpe_reward = calculate_sharpe_ratio(np.array(self.portfolio_return_history))
        # Apply a penalty for taking a buy/sell action to discourage over-trading
        penalty = self.action_penalty if action in [1, 2] else 0
        return sharpe_reward - penalty

    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, 'close']
        
        if action == 1 and self.position == 0: # Execute Buy
            self.position = 1
            self.entry_price = current_price
            # The balance at the start of the trade is adjusted for commission
            self.trade_start_balance = self.portfolio_value * (1 - self.commission_pct)
            self.trade_count += 1

        elif action == 2 and self.position == 1: # Execute Sell (to close long position)
            exit_price = current_price
            pnl = (exit_price - self.entry_price)
            # Update portfolio value with PnL from the starting balance
            self.portfolio_value = self.trade_start_balance + pnl
            # Apply commission on closing the trade
            self.portfolio_value *= (1 - self.commission_pct)
            
            # Reset position state
            self.position = 0
            self.entry_price = 0
            
    def _next_observation(self):
        # Get the sequence of data for the observation
        start_idx = self.current_step - self.sequence_length + 1
        end_idx = self.current_step
        
        features_df = self.df.loc[start_idx:end_idx, self.feature_cols]
        
        # Add the current position as a feature
        position_feature = np.zeros((len(features_df), 1))
        if self.position == 1:
            position_feature.fill(1)
            
        # Combine technical features and position feature
        obs = np.concatenate([features_df.values, position_feature], axis=1)
            
        return obs.astype(np.float32)