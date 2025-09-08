# envs/trading_env.py
# VERSION 7: CE-STRATEGIC ALIGNMENT WITH INACTIVITY PENALTY

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from utils.logger import get_logger
from config.init import Config

logger = get_logger(__name__)

class TradingEnv(gym.Env):
    """
    An advanced trading environment designed to teach an agent to strategically
    follow the Chandelier Exit (CE) indicator. The reward system is engineered
    to promote patience, strategic alignment, and robust risk management, while
    penalizing premature exits, hesitation, and general inactivity.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame):
        super(TradingEnv, self).__init__()

        # --- Data Validation and Initialization ---
        required_columns = ['close', 'ce_direction', 'bullish_flip', 'bearish_flip', 'dist_to_ce_stop_pct']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame is missing required column: {col}")

        self.df = df
        self.feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
        self.features_array = df[self.feature_cols].values.astype(np.float32)
        self.price_array = df['close'].values.astype(np.float32)
        self.ce_direction_array = df['ce_direction'].values.astype(np.int8)
        self.bullish_flip_array = df['bullish_flip'].values.astype(np.int8)
        self.bearish_flip_array = df['bearish_flip'].values.astype(np.int8)
        self.dist_to_ce_stop_pct_array = df['dist_to_ce_stop_pct'].values.astype(np.float32)

        self.max_steps = len(df) - 1

        # --- Load Configuration from YAML ---
        agent_config = Config()
        self.initial_balance = agent_config.get('environment.initial_balance', 10000)
        self.sequence_length = agent_config.get('environment.sequence_length', 70)
        self.commission_pct = agent_config.get('environment.commission_pct', 0.0005)
        self.max_account_drawdown = agent_config.get('environment.max_account_drawdown', 0.20)

        # --- Load Reward Shaping Parameters ---
        cfg_rewards = agent_config.get('environment.reward_shaping', {})
        self.patience_reward_factor = cfg_rewards.get('patience_reward_factor', 0.1)
        self.cowardice_penalty_factor = cfg_rewards.get('cowardice_penalty_factor', 50.0)
        self.profit_risk_reward_factor = cfg_rewards.get('profit_risk_reward_factor', 2.0)
        self.opportunity_cost_penalty_factor = cfg_rewards.get('opportunity_cost_penalty_factor', 0.5)
        self.turnover_penalty_factor = cfg_rewards.get('turnover_penalty_factor', 0.1)
        self.inactivity_penalty_factor = cfg_rewards.get('inactivity_penalty_factor', 0.01)
        self.max_idle_steps = cfg_rewards.get('max_idle_steps', 96) # e.g., one full day of 15min bars

        self.risk_proximity_penalty_enabled = cfg_rewards.get('risk_proximity_penalty_enabled', False)
        self.risk_proximity_penalty_factor = cfg_rewards.get('risk_proximity_penalty_factor', 1.0)

        # --- Action & Observation Spaces ---
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        obs_dim = self.features_array.shape[1] + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.sequence_length, obs_dim), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.sequence_length
        self.balance = self.initial_balance
        self.position_size = 0.0
        self.units_held = 0.0
        self.entry_price = 0.0
        self.high_water_mark = self.initial_balance
        
        self.trade_start_step = 0
        self.in_trade_high_water_mark = self.initial_balance
        self.in_trade_max_drawdown = 0.0
        self.opportunity_cost_counter = 0
        self.steps_since_last_trade = 0

        return self._next_observation(), {}

    def step(self, action):
        target_position_size = float(action[0])
        previous_position_size = self.position_size
        
        trade_executed = self._take_action(target_position_size)
        
        if trade_executed:
            self.steps_since_last_trade = 0
        else:
            self.steps_since_last_trade += 1

        self.current_step += 1
        current_equity = self._calculate_equity()
        is_trade_closed = (previous_position_size != 0 and self.position_size == 0)
        
        reward = self._get_reward(previous_position_size, is_trade_closed, current_equity, trade_executed)

        if self.position_size != 0:
            self.in_trade_high_water_mark = max(self.in_trade_high_water_mark, current_equity)
            drawdown = (self.in_trade_high_water_mark - current_equity) / (self.in_trade_high_water_mark + 1e-9)
            self.in_trade_max_drawdown = max(self.in_trade_max_drawdown, drawdown)
        
        self.high_water_mark = max(self.high_water_mark, current_equity)
        global_drawdown = (self.high_water_mark - current_equity) / (self.high_water_mark + 1e-9)

        done = (self.current_step >= self.max_steps)
        if global_drawdown > self.max_account_drawdown:
            done = True
            reward -= 200
            logger.warning(f"CRITICAL: Max account drawdown exceeded ({global_drawdown:.2%}). Terminating.")

        obs = self._next_observation()
        info = {'equity': current_equity, 'position_size': self.position_size}
        return obs, reward, done, False, info

    def _get_reward(self, previous_position_size, is_trade_closed, current_equity, trade_executed):
        reward = 0.0
        current_ce_direction = self.ce_direction_array[self.current_step]

        if is_trade_closed:
            profit = self.balance - self.entry_balance # Use balance at entry for accurate profit calculation
            
            if profit > 0:
                risk_adjusted_profit = profit / (self.in_trade_max_drawdown + 0.01)
                reward += risk_adjusted_profit * self.profit_risk_reward_factor

            exited_direction = np.sign(previous_position_size)
            if profit > 0 and exited_direction == self.ce_direction_array[self.current_step - 1]:
                 reward -= self.cowardice_penalty_factor
        
        elif self.position_size != 0: # In an open trade
            agent_direction = np.sign(self.position_size)
            if agent_direction == current_ce_direction:
                reward += self.patience_reward_factor
            
            if self.risk_proximity_penalty_enabled:
                dist_to_stop = self.dist_to_ce_stop_pct_array[self.current_step]
                if dist_to_stop < 1.0:
                    reward -= (1.0 - dist_to_stop) * self.risk_proximity_penalty_factor
            
        else: # Not in a trade
            is_bullish_flip = self.bullish_flip_array[self.current_step] == 1
            is_bearish_flip = self.bearish_flip_array[self.current_step] == 1
            if is_bullish_flip or is_bearish_flip:
                self.opportunity_cost_counter += 1
                if self.opportunity_cost_counter > 3:
                    reward -= self.opportunity_cost_penalty_factor
            
            # NEW: General inactivity penalty
            if self.steps_since_last_trade > self.max_idle_steps:
                reward -= self.inactivity_penalty_factor

        if trade_executed:
            reward -= self.turnover_penalty_factor

        return float(reward) if np.isfinite(reward) else -1.0

    def _take_action(self, target_position_size):
        current_price = self.price_array[self.current_step]
        trade_executed = False

        should_close = (self.position_size > 0 and target_position_size < 0.1) or \
                       (self.position_size < 0 and target_position_size > -0.1)

        if should_close and self.position_size != 0:
            realized_pnl = (current_price - self.entry_price) * self.units_held
            commission_cost = abs(self.units_held * current_price) * self.commission_pct
            self.balance += realized_pnl - commission_cost
            self.position_size, self.units_held, self.entry_price = 0.0, 0.0, 0.0
            
        elif abs(target_position_size) > 1e-6 and not should_close: # Open or increase position
            if self.position_size == 0.0: # Opening a new trade
                self.entry_price = current_price
                self.entry_balance = self.balance # Store balance at entry for PnL
                self.in_trade_high_water_mark = self.balance
                self.in_trade_max_drawdown = 0.0
                self.opportunity_cost_counter = 0
                trade_executed = True

            trade_value = (target_position_size - self.position_size) * self.balance
            commission_cost = abs(trade_value) * self.commission_pct
            self.balance -= commission_cost
            self.units_held += trade_value / current_price
            self.position_size = target_position_size
        
        return trade_executed

    def _calculate_equity(self):
        if self.position_size == 0.0:
            return self.balance
        unrealized_pnl = (self.price_array[self.current_step] - self.entry_price) * self.units_held
        return self.balance + unrealized_pnl

    def _next_observation(self):
        start_idx = self.current_step - self.sequence_length + 1
        end_idx = self.current_step + 1
        features = self.features_array[start_idx:end_idx]
        
        unrealized_pnl_pct = 0.0
        if self.position_size != 0:
            entry_value = abs(self.units_held * self.entry_price)
            if entry_value > 1e-6:
                unrealized_pnl = (self.price_array[self.current_step] - self.entry_price) * self.units_held
                unrealized_pnl_pct = unrealized_pnl / entry_value

        position_feature = np.full((self.sequence_length, 1), self.position_size)
        pnl_feature = np.full((self.sequence_length, 1), unrealized_pnl_pct)
        
        return np.concatenate([features, position_feature, pnl_feature], axis=1).astype(np.float32)