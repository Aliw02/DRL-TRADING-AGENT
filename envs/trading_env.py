# envs/trading_env.py (ADVANCED VERSION WITH TRUE CONTINUOUS ACTIONS)

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
    An advanced trading environment for SAC with a true continuous action space.
    The agent's action now represents the target portfolio allocation, allowing for
    variable position sizing and realistic portfolio management.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, initial_balance=None, sequence_length=None):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1

        self.initial_balance = initial_balance or config.get('environment.initial_balance', 10000)
        self.sequence_length = sequence_length or config.get('environment.sequence_length', 150)
        self.commission_pct = config.get('environment.commission_pct', 0.0005)
        self.turnover_penalty = config.get('environment.turnover_penalty', 0.001) # New penalty

        # ACTION SPACE: A single continuous value between -1 (100% Short) and +1 (100% Long).
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # OBSERVATION SPACE: We add the current position size as a feature.
        self.feature_cols = [col for col in self.df.columns if col not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
        obs_dim = len(self.feature_cols) + 1 # +1 for the current position size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.sequence_length, obs_dim), dtype=np.float32)
        
        logger.info(f"Advanced Environment initialized. Action space: Continuous (Allocation), Obs dim: {obs_dim}")
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.sequence_length
        
        # --- NEW STATE VARIABLES ---
        self.portfolio_value = self.initial_balance
        self.cash = self.initial_balance
        self.position_size = 0.0  # Represents the current allocation, e.g., 0.5 for 50% long
        self.units_held = 0.0     # Number of asset units held
        self.entry_price = 0.0
        
        self.trade_count = 0
        self.portfolio_return_history = [0.0] * self.sequence_length 
        
        return self._next_observation(), {}
    
    def step(self, action):
        target_position_size = action[0] # The agent's desired allocation (e.g., 0.8)
        
        prev_portfolio_value = self.portfolio_value
        
        # Calculate how much our position size changed (turnover)
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
        info = {'equity': self.portfolio_value, 'position_size': self.position_size, 'trade_count': self.trade_count}
        
        return obs, reward, done, False, info

    def _update_portfolio_value(self):
        # The portfolio value is the sum of cash and the market value of the current position.
        current_price = self.df.loc[self.current_step, 'close']
        position_market_value = self.units_held * current_price
        self.portfolio_value = self.cash + position_market_value

    def _get_reward(self, turnover):
        # The primary reward is still the Sharpe Ratio
        sharpe_reward = calculate_sharpe_ratio(np.array(self.portfolio_return_history))
        # We add a penalty for high turnover to discourage excessive trading
        penalty = self.turnover_penalty * turnover
        return sharpe_reward - penalty

    def _take_action(self, target_position_size):
        current_price = self.df.loc[self.current_step, 'close']
        
        # How many units we should hold for the target allocation
        target_units = (self.portfolio_value * target_position_size) / current_price
        
        # How many units we need to buy or sell to reach the target
        units_to_trade = target_units - self.units_held
        
        # --- EXECUTE THE TRADE ---
        if units_to_trade != 0:
            cost_of_trade = abs(units_to_trade * current_price)
            commission = cost_of_trade * self.commission_pct
            
            # Update cash, units held, and position size
            self.cash -= (units_to_trade * current_price) + commission
            self.units_held += units_to_trade
            self.position_size = (self.units_held * current_price) / self.portfolio_value if self.portfolio_value != 0 else 0
            
            # If we traded, count it
            self.trade_count += 1
            
    def _next_observation(self):
        start_idx = self.current_step - self.sequence_length + 1
        end_idx = self.current_step
        
        features_df = self.df.loc[start_idx:end_idx, self.feature_cols]
        
        # Add the current position size as the last feature in the observation
        position_feature = np.full((len(features_df), 1), self.position_size)
        
        obs = np.concatenate([features_df.values, position_feature], axis=1)
            
        return obs.astype(np.float32)