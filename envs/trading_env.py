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
        
        # --- NEW: Reward Shaping Parameters ---
        self.action_penalty = config.get('environment.action_penalty', 0.1) # Penalty for taking an action
        self.profit_bonus_multiplier = config.get('environment.profit_bonus', 2.0) # Bonus for closing a profitable trade

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
        
        # Execute the action and get any realized profit/loss
        realized_pnl = self._take_action(action)
        
        # Update the portfolio value based on market movement (unrealized PnL)
        self._update_portfolio_value()
        
        # Calculate the final reward for this step
        reward = self._get_reward(prev_portfolio_value, realized_pnl)
        
        self.current_step += 1
        
        done = (self.portfolio_value < self.initial_balance * 0.5) or (self.current_step >= self.max_steps)

        obs = self._next_observation()
        info = {'equity': self.portfolio_value, 'trade_count': self.trade_count}
        
        return obs, reward, done, False, info

    def _update_portfolio_value(self):
        """If a position is open, update the portfolio value based on the current price."""
        if self.position == 1:
            current_price = self.df.loc[self.current_step, 'close']
            unrealized_pnl = (current_price - self.entry_price)
            self.portfolio_value = self.trade_start_balance + unrealized_pnl

    def _get_reward(self, prev_portfolio_value, realized_pnl):
        """
        Calculates a "Guiding" reward with three components:
        1. Holding Reward: Based on the change in portfolio value (unrealized PnL).
        2. Realized Reward: A large bonus for closing a profitable trade.
        3. Action Penalty: A small cost for opening a trade.
        """
        # 1. Holding Reward (change in unrealized PnL)
        holding_reward = self.portfolio_value - prev_portfolio_value
        
        # 2. Realized Reward (bonus for taking profit)
        profit_bonus = 0
        if realized_pnl > 0:
            profit_bonus = realized_pnl * self.profit_bonus_multiplier
            
        # The total reward is the sum of all components
        reward = holding_reward + profit_bonus + realized_pnl # Add realized pnl directly
        
        return reward

    def _take_action(self, action):
        """Executes an action and returns the realized PnL for that step."""
        current_price = self.df.loc[self.current_step, 'close']
        realized_pnl = 0
        
        if action == 1 and self.position == 0: # Open a new long position
            self.position = 1
            self.entry_price = current_price # Slippage applied on close
            self.trade_start_balance = self.portfolio_value
            self.trade_count += 1
            # Apply action penalty as a cost
            self.portfolio_value -= self.action_penalty

        elif action == 2 and self.position == 1: # Close an existing long position
            exit_price = current_price
            realized_pnl = (exit_price - self.entry_price)
            
            # Apply slippage and commission as a percentage of the trade value
            trade_value = self.trade_start_balance
            commission = trade_value * self.commission_pct
            slippage = trade_value * self.slippage_pct # Simplified slippage
            
            self.portfolio_value += realized_pnl - commission - slippage
            
            self.position = 0
            self.entry_price = 0
            
        return realized_pnl
            
    def _next_observation(self):
        start_idx = max(0, self.current_step - self.sequence_length + 1)
        end_idx = self.current_step + 1
        
        features_df = self.df.loc[start_idx:end_idx, self.feature_cols]
        
        position_feature = np.zeros((len(features_df), 1))
        if self.position == 1:
            position_feature.fill(1)
            
        obs = np.concatenate([features_df.values, position_feature], axis=1)

        if len(obs) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(obs), obs.shape[1]))
            obs = np.vstack([padding, obs])
            
        return obs.astype(np.float32)

