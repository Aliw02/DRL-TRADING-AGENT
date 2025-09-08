# envs/trading_env.py
# --- VERSION 5: HIERARCHICAL REWARD - DAILY CONSISTENCY & MONTHLY AMBITION ---

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from utils.logger import get_logger
from config.init import Config

logger = get_logger(__name__)

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame):
        super(TradingEnv, self).__init__()

        self.feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
        self.features_array = df[self.feature_cols].values.astype(np.float32)
        self.price_array = df['close'].values.astype(np.float32)
        self.max_steps = len(df) - 1

        agent_config = Config()
        self.initial_balance = agent_config.get('environment.initial_balance', 10000)
        self.sequence_length = agent_config.get('environment.sequence_length', 70)
        self.commission_pct = agent_config.get('environment.commission_pct', 0.0005)
        self.max_account_drawdown = agent_config.get('environment.max_account_drawdown', 0.20)

        # --- 💡 تحميل جميع إعدادات المكافآت الهرمية ---
        self.daily_growth_reward_factor = agent_config.get('environment.daily_growth_reward_factor', 50.0)
        self.volatility_penalty_factor = agent_config.get('environment.volatility_penalty_factor', 0.5)
        self.steps_per_day = 96

        self.monthly_profit_target = agent_config.get('environment.monthly_profit_target_pct', 0.50)
        self.monthly_bonus = agent_config.get('environment.monthly_bonus_reward', 300.0)
        self.steps_per_month = agent_config.get('environment.steps_per_month', 2016)

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        obs_dim = self.features_array.shape[1] + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.sequence_length, obs_dim), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.sequence_length
        self.balance = self.initial_balance
        # ... (باقي متغيرات reset تبقى كما هي) ...
        self.entry_price = 0.0
        self.position_size = 0.0
        self.units_held = 0.0
        self.high_water_mark = self.initial_balance
        self.previous_step_equity = self.initial_balance
        self.day_start_step = self.current_step
        self.daily_high_water_mark = self.initial_balance
        
        # --- إعادة تعيين متغيرات الشهر ---
        self.month_start_step = self.current_step
        self.previous_month_end_balance = self.initial_balance

        return self._next_observation(self.initial_balance), {}

    def step(self, action):
        # ... (دالة step تبقى كما هي) ...
        target_position_size = float(action[0])
        self._take_action(target_position_size)
        self.current_step += 1
        current_equity = self._calculate_equity()
        reward = self._get_reward(current_equity)
        self.high_water_mark = max(self.high_water_mark, current_equity)
        drawdown = (self.high_water_mark - current_equity) / self.high_water_mark
        done = (self.current_step >= self.max_steps)
        if drawdown > self.max_account_drawdown:
            done = True
            reward -= 70
            logger.warning(f"CRITICAL: Max drawdown exceeded ({drawdown:.2%}). Terminating episode.")
        self.previous_step_equity = current_equity
        obs = self._next_observation(current_equity)
        info = {'equity': current_equity}
        return obs, reward, done, False, info
        
    def _calculate_equity(self):
        # ... (دالة _calculate_equity تبقى كما هي) ...
        unrealized_pnl = 0
        if self.position_size != 0:
            current_price = self.price_array[self.current_step]
            unrealized_pnl = (current_price - self.entry_price) * self.units_held
        return self.balance + unrealized_pnl

    def _get_reward(self, current_equity):
        """
        V5 Reward Function: Hierarchical rewards combining daily consistency
        with ambitious monthly targets.
        """
        # 1. المكافأة الأساسية اللحظية
        core_reward = np.log(current_equity / (self.previous_step_equity + 1e-9)) * 100

        # 2. عقوبة التقلبات (الخسارة)
        volatility_penalty = 0.0
        equity_change = current_equity - self.previous_step_equity
        if equity_change < 0:
            volatility_penalty = abs(equity_change / (self.previous_step_equity + 1e-9)) * self.volatility_penalty_factor * -100

        # 3. المكافأة اليومية للنمو الحقيقي
        daily_growth_reward = 0.0
        if self.current_step >= self.day_start_step + self.steps_per_day:
            if current_equity > self.initial_balance and current_equity > self.daily_high_water_mark:
                growth_amount = np.log(current_equity / self.daily_high_water_mark)
                daily_growth_reward = growth_amount * self.daily_growth_reward_factor
                logger.info(f"🏆 REAL CUMULATIVE GROWTH! New HWM: {current_equity:.2f}. Reward: {daily_growth_reward:.2f}")
                self.daily_high_water_mark = current_equity
            self.day_start_step = self.current_step

        # --- 💡 4. المكافأة الشهرية الطموحة (تمت إعادتها) ---
        monthly_bonus_reward = 0.0
        if self.current_step >= self.month_start_step + self.steps_per_month:
            target_balance = self.previous_month_end_balance * (1 + self.monthly_profit_target)
            if self.balance > target_balance:
                monthly_bonus_reward = self.monthly_bonus
                logger.info(f"🚀🚀🚀 AMBITIOUS MONTHLY TARGET MET! Balance {self.balance:.2f} > Target {target_balance:.2f}. HUGE Bonus: {monthly_bonus_reward}")
                self.previous_month_end_balance = self.balance
            else:
                logger.warning(f"Monthly objective MISSED. Balance {self.balance:.2f} <= Target {target_balance:.2f}.")
            self.month_start_step = self.current_step

        # تجميع كل المكافآت والعقوبات
        total_reward = core_reward + volatility_penalty + daily_growth_reward + monthly_bonus_reward
        return float(total_reward) if np.isfinite(total_reward) else -1.0


    def _take_action(self, target_position_size):
        # (This function remains unchanged)
        # ...
        current_price = self.price_array[self.current_step]
        if (self.position_size > 0 and target_position_size <= 0.1) or \
           (self.position_size < 0 and target_position_size >= -0.1):
            if self.position_size != 0:
                realized_pnl = (current_price - self.entry_price) * self.units_held
                commission = abs(self.units_held * current_price) * self.commission_pct
                self.balance += realized_pnl - commission
                self.position_size, self.units_held, self.entry_price = 0.0, 0.0, 0.0
        elif abs(target_position_size) > abs(self.position_size):
            if self.position_size == 0.0: self.entry_price = current_price
            trade_value = (target_position_size - self.position_size) * self.balance
            commission = abs(trade_value) * self.commission_pct
            self.balance -= commission
            self.units_held += trade_value / current_price
            self.position_size = target_position_size

    def _next_observation(self, current_equity):
        # (This function remains unchanged)
        # ...
        unrealized_pnl_pct = 0
        if self.position_size != 0:
             entry_value = abs(self.units_held * self.entry_price)
             unrealized_pnl = (current_equity - self.balance) - (abs(self.units_held * current_equity) * self.commission_pct)
             unrealized_pnl_pct = unrealized_pnl / (entry_value + 1e-9)
        start_idx = self.current_step - self.sequence_length + 1
        end_idx = self.current_step + 1
        features = self.features_array[start_idx:end_idx]
        state_features = np.array([[self.position_size] * self.sequence_length, [unrealized_pnl_pct] * self.sequence_length]).T
        return np.concatenate([features, state_features], axis=1)