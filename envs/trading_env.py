# envs/trading_env.py
# --- VERSION 3: With Logarithmic Daily Rewards & Volatility Penalty ---

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

        # ... ( باقي كود __init__ يبقى كما هو ) ...
        self.features_array = df[[c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]].values.astype(np.float32)
        self.price_array = df['close'].values.astype(np.float32)
        self.max_steps = len(df) - 1

        agent_config = Config()
        self.initial_balance = agent_config.get('environment.initial_balance', 10000)
        self.sequence_length = agent_config.get('environment.sequence_length', 30)
        self.commission_pct = agent_config.get('environment.commission_pct', 0.0005)
        self.max_account_drawdown = agent_config.get('environment.max_account_drawdown', 0.20)
        
        self.monthly_profit_target = agent_config.get('environment.monthly_profit_target_pct', 0.15)
        self.monthly_bonus = agent_config.get('environment.monthly_bonus_reward', 150.0)
        self.steps_per_month = agent_config.get('environment.steps_per_month', 2016)

        # --- 💡 التعديل: استبدال المكافآت الثابتة بمعامل ضرب ---
        self.daily_reward_factor = agent_config.get('environment.daily_reward_factor', 25.0) # معامل لزيادة أهمية الربح اليومي
        self.volatility_penalty_factor = agent_config.get('environment.volatility_penalty_factor', 0.5)
        self.steps_per_day = 96

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        obs_dim = self.features_array.shape[1] + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.sequence_length, obs_dim), dtype=np.float32)
        self.reset()

    # ... ( دالة reset و step و _calculate_equity تبقى كما هي ) ...
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.sequence_length
        self.balance = self.initial_balance
        self.entry_price = 0.0
        self.position_size = 0.0
        self.units_held = 0.0
        self.high_water_mark = self.initial_balance
        self.previous_step_equity = self.initial_balance
        self.month_start_step = self.current_step
        self.previous_month_end_balance = self.initial_balance
        self.day_start_step = self.current_step
        self.previous_day_end_equity = self.initial_balance
        return self._next_observation(self.initial_balance), {}

    def step(self, action):
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
            reward -= 50
            logger.warning(f"CRITICAL: Max drawdown exceeded ({drawdown:.2%}). Terminating episode.")
        self.previous_step_equity = current_equity
        obs = self._next_observation(current_equity)
        info = {'equity': current_equity}
        return obs, reward, done, False, info

    def _calculate_equity(self):
        unrealized_pnl = 0
        if self.position_size != 0:
            current_price = self.price_array[self.current_step]
            unrealized_pnl = (current_price - self.entry_price) * self.units_held
        return self.balance + unrealized_pnl
        
    def _get_reward(self, current_equity):
        """
        V3 Reward Function: Logarithmic returns for core and daily rewards.
        """
        # 1. المكافأة الأساسية (اللحظية)
        core_reward = np.log(current_equity / (self.previous_step_equity + 1e-9)) * 100

        # 2. عقوبة التقلبات (الخسارة)
        volatility_penalty = 0.0
        equity_change = current_equity - self.previous_step_equity
        if equity_change < 0:
            volatility_penalty = abs(equity_change / (self.previous_step_equity + 1e-9)) * self.volatility_penalty_factor * -100

        # 3. المكافأة الشهرية
        monthly_bonus_reward = 0.0
        # ... (منطق المكافأة الشهرية يبقى كما هو) ...
        if self.current_step >= self.month_start_step + self.steps_per_month:
            target_balance = self.previous_month_end_balance * (1 + self.monthly_profit_target)
            if self.balance > target_balance:
                monthly_bonus_reward = self.monthly_bonus
                self.previous_month_end_balance = self.balance
            self.month_start_step = self.current_step

        # --- 💡 4. المكافأة اليومية اللوغاريتمية (التعديل الرئيسي) ---
        daily_log_reward = 0.0
        if self.current_step >= self.day_start_step + self.steps_per_day:
            daily_log_return = np.log(current_equity / (self.previous_day_end_equity + 1e-9))
            
            # نعطي المكافأة فقط إذا كان هناك ربح
            if daily_log_return > 0:
                daily_log_reward = daily_log_return * self.daily_reward_factor
                logger.info(f"📈 Daily logarithmic growth! Current Equity {current_equity:.2f} > Previous Equity: {self.previous_day_end_equity:.2f}. Reward: {daily_log_reward:.2f}")

            self.previous_day_end_equity = current_equity
            self.day_start_step = self.current_step

        total_reward = core_reward + volatility_penalty + monthly_bonus_reward + daily_log_reward
        return float(total_reward) if np.isfinite(total_reward) else -1.0

    # ... ( باقي الدوال _take_action و _next_observation تبقى كما هي ) ...
    def _take_action(self, target_position_size):
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