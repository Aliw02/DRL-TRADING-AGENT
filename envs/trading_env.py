# envs/trading_env.py
# FINAL, OBJECTIVE-DRIVEN, AND STRATEGICALLY-AWARE TRADING ENVIRONMENT

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
        
        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1
        
        # --- Load strategic parameters from the master configuration ---
        self.initial_balance = config.get('environment.initial_balance', 10000)
        self.sequence_length = config.get('environment.sequence_length', 30)
        self.commission_pct = config.get('environment.commission_pct', 0.0005)
        
        # --- NEW: Load strategic objective parameters ---
        self.monthly_profit_target = config.get('environment.monthly_profit_target_pct', 0.15)
        self.monthly_bonus = config.get('environment.monthly_bonus_reward', 150.0)
        self.steps_per_month = config.get('environment.steps_per_month', 2016)
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        self.feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
        obs_dim = self.features_array.shape[1] + 2 # Features + position_size, unrealized_pnl_pct
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.sequence_length, obs_dim), 
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.sequence_length
        self.balance = self.initial_balance
        self.entry_price = 0.0
        self.position_size = 0.0
        self.units_held = 0.0
        self.unrealized_pnl_pct = 0.0
        
        # --- NEW: Initialize trackers for the monthly bonus ---
        self.month_start_step = self.current_step
        self.month_start_balance = self.initial_balance
        
        return self._next_observation(), {}
    
    def step(self, action):
        target_position_size = float(action[0])
        
        realized_pnl = self._take_action(target_position_size)
        
        self.current_step += 1
        self._update_unrealized_pnl()
        
        reward = self._get_reward(realized_pnl)
        
        done = (self.balance < self.initial_balance * 0.5) or (self.current_step >= self.max_steps)
        obs = self._next_observation()
        info = {'equity': self.balance + (self.unrealized_pnl_pct * self.entry_price * abs(self.units_held))}
        
        return obs, reward, done, False, info

    def _update_unrealized_pnl(self):
        # ... (This function remains unchanged)
        if self.position_size != 0:
            current_price = self.price_array[self.current_step]
            pnl = (current_price - self.entry_price) * self.position_size
            self.unrealized_pnl_pct = (pnl / self.entry_price) if self.entry_price > 1e-9 else 0.0
        else:
            self.unrealized_pnl_pct = 0.0

    def _get_reward(self, realized_pnl):
        """
        OBJECTIVE-DRIVEN REWARD DOCTRINE.
        Includes a decisive bonus for meeting long-term performance targets.
        """
        # --- 1. CONTINUOUS PROFIT/LOSS (Shaping Reward) ---
        shaping_reward = np.tanh(self.unrealized_pnl_pct * 10)
        
        # --- 2. DECISIVE OUTCOME (Event Reward) ---
        realized_reward = (realized_pnl / self.initial_balance) * 100
        
        # --- 3. STRATEGIC OBJECTIVE (Performance Bonus) ---
        bonus_reward = 0.0
        # Check if a month has passed
        if self.current_step >= self.month_start_step + self.steps_per_month:
            # Check if the profit target was met
            profit_pct = (self.balance / self.month_start_balance) - 1
            if profit_pct >= self.monthly_profit_target:
                bonus_reward = self.monthly_bonus
                logger.info(f"🏆 STRATEGIC OBJECTIVE MET! Awarding bonus of {bonus_reward}")
            
            # Reset the trackers for the next month
            self.month_start_step = self.current_step
            self.month_start_balance = self.balance
        
        # --- FINAL REWARD: The sum of all reward components ---
        reward = shaping_reward + realized_reward + bonus_reward
        
        return float(reward) if np.isfinite(reward) else -10.0

    def _take_action(self, target_position_size):
        # ... (This function remains unchanged)
        current_price = self.price_array[self.current_step]
        realized_pnl = 0.0
        if (self.position_size > 0 and target_position_size <= 0) or \
           (self.position_size < 0 and target_position_size >= 0):
            final_value = (self.units_held * current_price)
            realized_pnl = final_value - (self.entry_price * self.units_held)
            commission = abs(final_value) * self.commission_pct
            self.balance += realized_pnl - commission
            self.position_size, self.units_held, self.entry_price = 0.0, 0.0, 0.0
        elif abs(target_position_size) > abs(self.position_size) and abs(target_position_size) > 0.1:
            if self.position_size == 0.0: self.entry_price = current_price
            trade_value = self.balance * (target_position_size - self.position_size)
            commission = abs(trade_value) * self.commission_pct
            self.balance -= commission
            self.units_held += trade_value / current_price
            self.position_size = (self.units_held * current_price) / self.balance if self.balance > 1e-9 else 0.0
        return realized_pnl
            
    def _next_observation(self):
        # ... (This function remains unchanged)
        start_idx = self.current_step - self.sequence_length + 1
        end_idx = self.current_step + 1
        features = self.features_array[start_idx:end_idx]
        state_features = np.array([
            [self.position_size] * self.sequence_length,
            [self.unrealized_pnl_pct] * self.sequence_length
        ]).T
        return np.concatenate([features, state_features], axis=1)