# envs/trading_env.py
# STRATEGICALLY RE-ENGINEERED ENVIRONMENT
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from utils.logger import get_logger
from config.init import config
from utils.metrics import calculate_sortino_ratio

logger = get_logger(__name__)

class TradingEnv(gym.Env):
    """
    An advanced trading environment engineered to teach a doctrine of
    risk-adjusted returns and capital preservation. It utilizes a sophisticated
    hybrid reward function that balances Sortino Ratio, drawdown penalties,
    and profit-seeking behavior.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, is_eval_env: bool = False):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1

        # Load strategic parameters from the master configuration
        self.initial_balance = config.get('environment.initial_balance', 10000)
        self.sequence_length = config.get('environment.sequence_length', 60)
        self.commission_pct = config.get('environment.commission_pct', 0.0005)
        self.turnover_penalty_weight = config.get('environment.turnover_penalty', 0.001)

        # The action space is a continuous value representing the desired portfolio allocation
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Define feature columns dynamically, excluding price and time information
        self.feature_cols = [col for col in self.df.columns if col not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
        # Observation space includes all market features plus the current position size
        obs_dim = len(self.feature_cols) + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.sequence_length, obs_dim),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.sequence_length
        self.portfolio_value = self.initial_balance
        self.cash = self.initial_balance
        self.position_size = 0.0 # Current portfolio allocation (-1 to 1)
        self.units_held = 0.0 # Number of asset units held

        # Initialize trackers for the hybrid reward function
        self.portfolio_return_history = [0.0] * self.sequence_length
        self.max_portfolio_value = self.initial_balance

        return self._next_observation(), {}

    def step(self, action):
        target_position_size = float(action[0])
        prev_portfolio_value = self.portfolio_value

        # Calculate turnover (how much the agent changed its position)
        turnover = abs(target_position_size - self.position_size)

        # Execute the trade based on the new target position
        self._take_action(target_position_size)

        # Update the portfolio's market value
        self._update_portfolio_value()

        # Calculate the return for this step and update history
        step_return = (self.portfolio_value / prev_portfolio_value) - 1 if prev_portfolio_value > 1e-9 else 0
        self.portfolio_return_history.append(step_return)
        self.portfolio_return_history.pop(0) # Maintain rolling window

        # Calculate the reward using our advanced doctrine
        reward = self._get_reward(turnover)

        self.current_step += 1

        # Define termination conditions
        # 1. Catastrophic loss (e.g., >50% drawdown)
        # 2. Reached the end of the dataset
        done = (self.portfolio_value < self.initial_balance * 0.5) or (self.current_step >= self.max_steps)

        obs = self._next_observation()
        info = {'equity': self.portfolio_value, 'position_size': self.position_size}

        return obs, reward, done, False, info

    def _update_portfolio_value(self):
        current_price = self.df.loc[self.current_step, 'close']
        position_market_value = self.units_held * current_price
        self.portfolio_value = self.cash + position_market_value

        # Update the high-water mark for drawdown calculation
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value

    def _get_reward(self, turnover):
        """
        The Hybrid Reward Engine. This function enforces our strategic doctrine.
        """
        # --- Offensive Component: Risk-Adjusted Return (Sortino Ratio) ---
        # We use a numpy array for efficient calculation
        returns_np = np.array(self.portfolio_return_history)
        sortino_reward = calculate_sortino_ratio(returns_np)

        # --- Tactical Component: Penalize for hesitation or small gains ---
        # We use tanh to squash the return, rewarding strong moves more
        pnl_reward = np.tanh(returns_np[-1] * 10)

        # --- Defensive Component: Max Drawdown Penalty ---
        # The penalty is squared to punish large drawdowns exponentially
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        drawdown_penalty = -(current_drawdown ** 2)

        # --- Efficiency Component: Turnover Penalty ---
        # Punishes the agent for excessive trading, simulating commission/slippage
        turnover_penalty = self.turnover_penalty_weight * turnover

        # --- Final Weighted Reward ---
        # The weights determine the agent's overall behavior (risk-averse vs. profit-seeking)
        reward = (
            (sortino_reward * 0.5) +  # Primary objective
            (pnl_reward * 0.1) +      # Tactical incentive
            (drawdown_penalty * 0.4) -# Defensive constraint
            turnover_penalty         # Efficiency constraint
        )

        # Ensure reward is a valid number
        return 0.0 if np.isnan(reward) else float(reward)

    def _take_action(self, target_position_size):
        current_price = self.df.loc[self.current_step, 'close']
        if current_price <= 1e-9: return # Avoid division by zero

        # Calculate the target value in the asset
        target_asset_value = self.portfolio_value * target_position_size
        target_units = target_asset_value / current_price

        # Calculate the change in units required
        units_to_trade = target_units - self.units_held
        
        if abs(units_to_trade) > 1e-6:
            cost_of_trade = abs(units_to_trade * current_price)
            commission = cost_of_trade * self.commission_pct

            # Update cash and units held
            self.cash -= (units_to_trade * current_price) + commission
            self.units_held += units_to_trade

        # Update the official position size after the trade
        self.position_size = (self.units_held * current_price) / self.portfolio_value if self.portfolio_value > 1e-9 else 0

    def _next_observation(self):
        # Get the sequence of market data
        start_idx = self.current_step - self.sequence_length + 1
        end_idx = self.current_step
        features_df = self.df.loc[start_idx:end_idx, self.feature_cols]

        # Add the current position size as the final feature
        position_feature = np.full((len(features_df), 1), self.position_size, dtype=np.float32)

        # Combine market features and position feature
        obs = np.concatenate([features_df.values, position_feature], axis=1)
        return obs.astype(np.float32)