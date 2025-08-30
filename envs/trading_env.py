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
        self.slippage_pct = config.get('environment.slippage_pct', 0.0002)

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
        self.portfolio_value = self.initial_balance # Use a single variable for account value
        self.trade_count = 0
        
        self.max_portfolio_value = self.initial_balance
        self.equity_history = [self.initial_balance] * 50
        
        return self._next_observation(), {}
    
    def step(self, action):
        prev_portfolio_value = self.portfolio_value
        
        self._take_action(action)
        self._update_portfolio_value()
        
        self.equity_history.append(self.portfolio_value)
        self.equity_history.pop(0)
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)

        reward = self._get_reward(prev_portfolio_value)
        
        self.current_step += 1
        
        # --- ROBUST DONE CONDITION ---
        # End episode if value drops below 50% OR if we reach the end of data
        done = (self.portfolio_value < self.initial_balance * 0.5) or (self.current_step >= self.max_steps)

        obs = self._next_observation()
        info = {'equity': self.portfolio_value, 'trade_count': self.trade_count}
        
        return obs, reward, done, False, info

    def _update_portfolio_value(self):
        """If a position is open, update the portfolio value based on the current price."""
        if self.position == 1:
            current_price = self.df.loc[self.current_step, 'close']
            unrealized_pnl = (current_price - self.entry_price)
            # The portfolio value is the starting balance of the trade + unrealized PnL
            self.portfolio_value = self.trade_start_balance + unrealized_pnl

    def _get_reward(self, prev_portfolio_value):
        """Calculates a more stable multi-objective reward."""
        pnl_reward = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        drawdown_penalty = - (current_drawdown ** 2)
        
        returns = pd.Series(self.equity_history).pct_change().dropna()
        negative_returns = returns[returns < 0]
        
        sortino_ratio = (returns.mean() / negative_returns.std()) * 0.1 if len(negative_returns) > 1 and negative_returns.std() != 0 else 0
            
        reward = (pnl_reward * 100) + (drawdown_penalty * 2.0) + sortino_ratio
        
        return reward if np.isfinite(reward) else 0.0

    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, 'close']
        
        if action == 1 and self.position == 0: # Open a new long position
            self.position = 1
            self.entry_price = current_price * (1 + self.slippage_pct)
            self.trade_start_balance = self.portfolio_value # Store balance when trade opens
            self.portfolio_value -= self.portfolio_value * self.commission_pct # Apply commission
            self.trade_count += 1

        elif action == 2 and self.position == 1: # Close an existing long position
            self.position = 0
            # Portfolio value is already updated in _update_portfolio_value, just apply commission
            self.portfolio_value -= self.portfolio_value * self.commission_pct
            self.entry_price = 0
            # Reset max value to "bank" the profits from the last trade
            self.max_portfolio_value = self.portfolio_value
            
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
