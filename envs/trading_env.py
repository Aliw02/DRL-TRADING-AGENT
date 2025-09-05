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
        
        # Load strategic parameters from the master configuration
        self.initial_balance = config.get('environment.initial_balance', 10000)
        self.sequence_length = config.get('environment.sequence_length', 30)
        self.commission_pct = config.get('environment.commission_pct', 0.0005) # Represents spread + commission
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
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
        self.position_size = 0.0 # -1 (Short) to 1 (Long)
        self.units_held = 0.0
        self.unrealized_pnl_pct = 0.0
        
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
        """ Calculates the unrealized Profit/Loss for the currently open position. """
        if self.position_size != 0:
            current_price = self.price_array[self.current_step]
            pnl = (current_price - self.entry_price) * self.position_size
            # We use percentage PnL to keep the reward signal stable
            self.unrealized_pnl_pct = (pnl / self.entry_price) if self.entry_price > 1e-9 else 0.0
        else:
            self.unrealized_pnl_pct = 0.0

    def _get_reward(self, realized_pnl):
        """
        THE COMMANDER'S DOCTRINE REWARD FUNCTION.
        This function implements the exact strategy you described.
        """
        # --- 1. CONTINUOUS PROFIT/LOSS (Shaping Reward) ---
        # At every step, the agent gets a small reward for holding a winner
        # and a small penalty for holding a loser. This teaches trend-following.
        # We use tanh to keep the signal bounded and stable.
        shaping_reward = np.tanh(self.unrealized_pnl_pct * 10)
        
        # --- 2. DECISIVE OUTCOME (Event Reward) ---
        # A large, decisive reward or penalty is given ONLY when a trade is closed.
        # This is the primary driver for learning profitability.
        # The reward is proportional to the percentage PnL.
        realized_reward = (realized_pnl / self.initial_balance) * 100 # Scaled for significance
        
        # --- FINAL REWARD: The sum of continuous guidance and decisive outcomes ---
        reward = shaping_reward + realized_reward
        
        # Safety net to prevent system instability
        return float(reward) if np.isfinite(reward) else -10.0 # Severe penalty for invalid states

    def _take_action(self, target_position_size):
        """ Executes trades and returns the REALIZED PnL from any closing action. """
        current_price = self.price_array[self.current_step]
        realized_pnl = 0.0

        # --- ACTION: CLOSE or REDUCE POSITION ---
        # This is triggered if we have a position and the agent wants to exit or flip.
        if (self.position_size > 0 and target_position_size < self.position_size) or \
           (self.position_size < 0 and target_position_size > self.position_size):
            
            # Close the existing position to realize PnL
            final_value = (self.units_held * current_price)
            realized_pnl = final_value - (self.entry_price * self.units_held)
            
            # Deduct commission from the cash balance
            commission = abs(final_value) * self.commission_pct
            self.balance += realized_pnl - commission
            
            # Reset position state
            self.position_size, self.units_held, self.entry_price = 0.0, 0.0, 0.0

        # --- ACTION: OPEN or INCREASE POSITION ---
        # This is triggered if we are not fully invested in the target direction.
        if abs(target_position_size) > abs(self.position_size) and abs(target_position_size) > 0.1:
            if self.position_size == 0.0: # New entry
                self.entry_price = current_price

            # Calculate the value to invest
            trade_value = self.balance * (target_position_size - self.position_size)
            commission = abs(trade_value) * self.commission_pct
            self.balance -= commission
            
            # Update units held
            self.units_held += trade_value / current_price
            self.position_size = (self.units_held * current_price) / self.balance if self.balance > 1e-9 else 0.0

        return realized_pnl
            
    def _next_observation(self):
        start_idx = self.current_step - self.sequence_length + 1
        end_idx = self.current_step + 1
        
        features = self.features_array[start_idx:end_idx]
        
        # Agent is now aware of its position and current trade performance
        state_features = np.array([
            [self.position_size] * self.sequence_length,
            [self.unrealized_pnl_pct] * self.sequence_length
        ]).T

        return np.concatenate([features, state_features], axis=1)