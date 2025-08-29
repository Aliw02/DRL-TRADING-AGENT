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
        
        self.initial_balance = initial_balance or config.get('environment.initial_balance', 10000)
        self.sequence_length = sequence_length or config.get('environment.sequence_length', 40)
        
        # استعادة تكاليف التداول
        self.commission_pct = config.get('environment.commission_pct', 0.0005)
        self.slippage_pct = config.get('environment.slippage_pct', 0.0002)

        # إضافة وزن المكافأة القائمة على المؤشر من ملف الإعدادات
        self.indicator_reward_weight = config.get('environment.indicator_reward_weight', 0.1)

        self.df = df.reset_index(drop=True)
        self.current_step = self.sequence_length
        self.max_steps = len(self.df) - 1
        
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        feature_df = self.df.drop(columns=['close'], errors='ignore')
        self.feature_count = feature_df.shape[1]
        obs_dim = self.feature_count + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.sequence_length, obs_dim),
            dtype=np.float32
        )
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.sequence_length
        self.position = 0          # 0: No Position, 1: Long Position
        self.entry_price = 0
        self.equity = self.initial_balance
        self.prev_equity = self.equity
        self.trade_count = 0
        return self._next_observation(), {}
    
    def step(self, action):
        # 1. احسب قيمة المحفظة الحالية قبل اتخاذ أي إجراء جديد
        self.prev_equity = self.equity
        current_price = self.df.loc[self.current_step, 'close']
        
        if self.position == 1:
            # قم بتحديث قيمة المحفظة بناءً على السعر الحالي
            self.equity = self.initial_balance + (current_price - self.entry_price)
        
        # 2. اتخذ الإجراء الجديد بناءً على الـ action
        self._take_action(action)
        
        # 3. احسب المكافأة باستخدام الطريقتين معاً
        reward = self._get_reward(action)
        
        # 4. قم بتحديث الخطوة
        self.current_step += 1
        
        done = self.current_step >= self.max_steps
        
        obs = self._next_observation()
        
        info = {'trade_count': self.trade_count, 'equity': self.equity, 'reward': reward}
        
        return obs, reward, done, False, info
        
    def _get_reward(self, action):
        current_price = self.df.loc[self.current_step, 'close']
        
        # 1. المكافأة الأساسية: الربح والخسارة المباشر من الصفقة
        # هذه المكافأة هي الأهم وتُعلم النموذج الهدف النهائي
        profit_loss = 0.0
        if self.position == 1:
            profit_loss = current_price - self.entry_price
        
        # 2. المكافأة المساعدة (Reward Shaping): لمساعدة النموذج على التعلم
        # هذه المكافأة تُعطى كـ "تلميح" على قرارات معينة
        indicator_reward = 0.0
        try:
            bullish_flip = self.df.loc[self.current_step, 'bullish_flip'] == 1
            bearish_flip = self.df.loc[self.current_step, 'bearish_flip'] == 1
            direction = self.df.loc[self.current_step, 'direction']
            
            # مكافأة صغيرة للاحتفاظ بصفقة في الاتجاه الصحيح
            if self.position == 1 and direction == 1:
                indicator_reward += 0.1 # مكافأة للحفاظ على صفقة شراء في اتجاه صاعد
            elif self.position == -1 and direction == -1:
                indicator_reward += 0.1 # مكافأة للحفاظ على صفقة بيع في اتجاه هابط

            # عقوبة للحفاظ على صفقة في الاتجاه الخاطئ
            elif self.position == 1 and direction == -1:
                indicator_reward -= 0.1 # عقوبة للحفاظ على صفقة شراء في اتجاه هابط
            elif self.position == -1 and direction == 1:
                indicator_reward -= 0.1 # عقوبة للحفاظ على صفقة بيع في اتجاه صاعد
            
            # مكافأة كبيرة عند فتح صفقة جديدة في الاتجاه الصحيح
            if (bullish_flip and action == 1) or (bearish_flip and action == 2):
                indicator_reward += 1.0
            
            # عقوبة على اتخاذ قرار خاطئ تماماً
            if (bullish_flip and action == 2) or (bearish_flip and action == 1):
                indicator_reward -= 1.0
                
        except Exception as e:
            logger.error(f"Error calculating indicator reward: {e}")

        # 3. دمج المكافآت معاً
        # مكافأة الربح والخسارة المالي هي الأساس، مع إضافة مكافأة المؤشر كدليل
        combined_reward = (profit_loss * 0.01) + (self.indicator_reward_weight * indicator_reward)
        return combined_reward
        
    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, 'close']
        
        if action == 1 and self.position == 0:
            # شراء: افتح مركز جديد
            self.position = 1
            self.entry_price = current_price * (1 + self.slippage_pct)
            self.trade_count += 1
            # طبق عمولة الشراء
            self.initial_balance -= self.initial_balance * self.commission_pct
            logger.info(f"Step {self.current_step}: BUY @ {current_price:.2f}. Entry Price: {self.entry_price:.2f}")

        elif action == 2 and self.position == 1:
            # بيع: أغلق المركز الحالي
            self.position = 0
            # احسب الربح/الخسارة وأضفه إلى رصيد الحساب
            profit_loss = (current_price - self.entry_price)
            self.initial_balance = self.initial_balance + profit_loss
            # طبق عمولة البيع
            self.initial_balance -= self.initial_balance * self.commission_pct
            self.entry_price = 0
            self.trade_count += 1
            logger.info(f"Step {self.current_step}: SELL @ {current_price:.2f}. Profit/Loss: {profit_loss:.2f}")

    def _next_observation(self):
        start_idx = max(0, self.current_step - self.sequence_length + 1)
        end_idx = self.current_step
        
        features_df = self.df.loc[start_idx:end_idx].drop(columns=['close'], errors='ignore')
        features = features_df.values
        
        position_feature = np.zeros((len(features), 1))
        if self.position == 1:
            position_feature.fill(1)
            
        obs = np.concatenate([features, position_feature], axis=1)

        if len(obs) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(obs), obs.shape[1]))
            obs = np.vstack([padding, obs])
            
        return obs.astype(np.float32)