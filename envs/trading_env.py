# envs/trading_env.py
# النسخة النهائية مع مكافأة مضخمة وغير متكافئة لتحفيز السعي نحو الربح

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from utils.logger import get_logger
from config.init import Config

logger = get_logger(__name__)

class TradingEnv(gym.Env):
    """
    بيئة تداول مع نظام مكافآت مضخم وغير متكافئ لتعليم النموذج
    السعي بنشاط لتحقيق الأرباح بدلاً من تجنب الخسارة فقط.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame):
        super(TradingEnv, self).__init__()

        # ... (كل أجزاء التهيئة الأخرى تبقى كما هي) ...
        required_columns = ['close', 'ce_direction', 'bullish_flip', 'bearish_flip']
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
        self.max_steps = len(df) - 1

        agent_config = Config()
        self.initial_balance = agent_config.get('environment.initial_balance', 10000)
        self.sequence_length = agent_config.get('environment.sequence_length', 70)
        self.commission_pct = agent_config.get('environment.commission_pct', 0.0005)
        self.max_account_drawdown = agent_config.get('environment.max_account_drawdown', 0.20)

        # =======================================================================
        # ========== بداية التعديل: تعريف المكافآت المضخمة ==========
        # =======================================================================
        self.profit_amplification_factor = 5.0 # تضخيم الأرباح 5 مرات
        self.loss_amplification_factor = 2.0   # تضخيم الخسائر مرتين فقط

        # بقية المكافآت والعقوبات تبقى كما هي
        self.milestone_bonus = 30.0
        self.milestone_percentage = 0.05
        self.late_entry_penalty_pct = 2.0
        self.inactivity_penalty_pct = 0.2
        # =======================================================================
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        obs_dim = self.features_array.shape[1] + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.sequence_length, obs_dim), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        # ... (دالة reset تبقى كما هي) ...
        super().reset(seed=seed)
        self.current_step = self.sequence_length
        self.balance = self.initial_balance
        self.position_size = 0.0
        self.units_held = 0.0
        self.entry_price = 0.0
        
        self.high_water_mark = self.initial_balance
        self.milestone_base = self.initial_balance
        self.next_milestone_target = self.milestone_base * (1 + self.milestone_percentage)
        
        self.previous_equity = self.initial_balance
        
        self.steps_since_flip = 0
        self.steps_since_last_trade = 0
        self.just_opened_trade = False
        self.entry_ce_direction = 0

        self.last_reported_equity = self.initial_balance
        self.steps_since_report = 0

        return self._next_observation(), {}

    def step(self, action):
        # ... (معظم دالة step تبقى كما هي) ...
        target_position_size = float(action[0])
        
        was_in_trade = (self.position_size != 0)
        
        if not was_in_trade and abs(target_position_size) > 0.1:
            self.entry_ce_direction = self.ce_direction_array[self.current_step]

        self._take_action(target_position_size)
        is_in_trade = (self.position_size != 0)
        
        self.just_opened_trade = (not was_in_trade and is_in_trade)
        just_closed_trade = (was_in_trade and not is_in_trade)

        self.current_step += 1
        if self.just_opened_trade or just_closed_trade:
            self.steps_since_last_trade = 0
        else:
            self.steps_since_last_trade += 1

        current_equity = self._calculate_equity()
        
        reward = self._get_reward(just_closed_trade, current_equity)

        self.previous_equity = current_equity

        self.high_water_mark = max(self.high_water_mark, current_equity)
        global_drawdown = (self.high_water_mark - current_equity) / (self.high_water_mark + 1e-9)
        
        done = (self.current_step >= self.max_steps)
        if global_drawdown > self.max_account_drawdown:
            done = True
            logger.warning(f"CRITICAL: Max account drawdown exceeded ({global_drawdown:.2%}). Terminating.")

        obs = self._next_observation()
        info = {'equity': current_equity, 'position_size': self.position_size}
        
        if self.steps_since_report >= 480:
            period_return = (current_equity / self.last_reported_equity - 1) * 100
            logger.info(
                f"🧠 Training Report | Step: {self.current_step} | "
                f"Current Equity: ${current_equity:,.2f} | "
                f"Period P/L: {period_return:+.2f}%"
            )
            self.last_reported_equity = current_equity
            self.steps_since_report = 0
        else:
            self.steps_since_report += 1

        return obs, reward, done, False, info

    def _get_reward(self, just_closed_trade, current_equity):
        """
        دالة مكافأة أساسها هو التغير اللحظي في الرصيد مع تضخيم غير متكافئ للأرباح.
        """
        equity_change = current_equity - self.previous_equity
        
        # --- 1. المكافأة/العقوبة الأساسية المضخمة ---
        if equity_change > 0:
            reward = equity_change * self.profit_amplification_factor
        else:
            reward = equity_change * self.loss_amplification_factor

        # --- 2. المكافأة الهرمية (إضافية) ---
        if current_equity > self.next_milestone_target:
            reward += self.milestone_bonus
            self.milestone_base = current_equity
            self.next_milestone_target = self.milestone_base * (1 + self.milestone_percentage)
            logger.info(f"🎉 MILESTONE REACHED! New Equity High: ${current_equity:,.2f}. Bonus: {self.milestone_bonus}. Next target: ${self.next_milestone_target:,.2f}")

        # --- 3. حافز إضافي للصفقات المتوافقة (عند الإغلاق فقط) ---
        if just_closed_trade:
            trade_was_aligned = (np.sign(self.last_trade_direction) == self.entry_ce_direction)
            if self.last_trade_profit > 0 and trade_was_aligned:
                reward += (self.last_trade_profit / self.initial_balance) * 10

        # --- 4. العقوبات السلوكية (للحفاظ على الانضباط) ---
        is_flip_signal = self.bullish_flip_array[self.current_step] == 1 or self.bearish_flip_array[self.current_step] == 1
        if is_flip_signal: self.steps_since_flip = 0
        else: self.steps_since_flip += 1
            
        if self.just_opened_trade and self.steps_since_flip > 2:
            reward -= self.late_entry_penalty_pct

        if self.position_size == 0 and self.steps_since_last_trade > 96 * 5:
            reward -= self.inactivity_penalty_pct
            
        return float(reward) if np.isfinite(reward) else -1.0


    # ... (باقي دوال الملف تبقى كما هي بدون تغيير) ...
    def _take_action(self, target_position_size):
        # ...
        current_price = self.price_array[self.current_step]
        previous_position_size = self.position_size

        should_close = (self.position_size > 0 and target_position_size < 0.1) or \
                       (self.position_size < 0 and target_position_size > -0.1)

        if should_close and self.position_size != 0:
            realized_pnl = (current_price - self.entry_price) * self.units_held
            commission_cost = abs(self.units_held * current_price) * self.commission_pct
            self.balance += realized_pnl - commission_cost
            
            self.last_trade_profit = realized_pnl - commission_cost
            self.last_trade_direction = previous_position_size
            
            self.position_size, self.units_held, self.entry_price = 0.0, 0.0, 0.0
            
        elif abs(target_position_size) > 1e-6 and not should_close:
            is_new_trade = (self.position_size == 0.0)
            if is_new_trade:
                self.entry_price = current_price

            trade_value = (target_position_size - self.position_size) * self.balance
            commission_cost = abs(trade_value) * self.commission_pct
            self.balance -= commission_cost
            self.units_held += trade_value / current_price
            self.position_size = target_position_size

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