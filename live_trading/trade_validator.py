# live_trading/trade_validator.py
# PRE-TRADE SIMULATION AND VALIDATION UNIT

import numpy as np
import pandas as pd
import sys
import os

# --- Add project root to path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from utils.logger import get_logger

logger = get_logger(__name__)

class TradeValidator:
    def __init__(self, historical_data: pd.DataFrame):
        """
        Initializes the validator with a recent chunk of historical data
        to calculate realistic market volatility.
        """
        self.historical_data = historical_data

    def _generate_future_scenarios(self, last_known_price, num_scenarios=100, num_steps=20):
        """
        Generates multiple possible future price paths using a simplified
        Monte Carlo simulation based on recent historical volatility.
        """
        log_returns = np.log(self.historical_data['close'] / self.historical_data['close'].shift(1))
        volatility = log_returns.std()
        if volatility == 0: volatility = 0.0001 # Avoid zero volatility

        scenarios = np.zeros((num_scenarios, num_steps))
        # Use a geometric Brownian motion model
        drift = log_returns.mean()
        for i in range(num_scenarios):
            shocks = np.random.normal(drift, volatility, num_steps)
            path = last_known_price * np.exp(np.cumsum(shocks))
            scenarios[i, :] = path
        
        return scenarios

    def validate_trade(self, proposed_action: float):
        """
        Runs a micro-backtest on the proposed trade to check its viability
        against a range of probable future scenarios.

        Returns:
            bool: True if the trade is validated, False otherwise.
        """
        logger.info(f"🔬 Validating proposed trade action: {proposed_action:.2f}...")
        
        last_real_price = self.historical_data['close'].iloc[-1]
        future_scenarios = self._generate_future_scenarios(last_real_price)
        
        final_pnls = []

        for scenario in future_scenarios:
            # For simplicity, we assume a fixed stop-loss and take-profit based on recent ATR
            atr = self.historical_data['atr'].iloc[-1]
            pnl = self._simulate_one_path(proposed_action, scenario, atr)
            final_pnls.append(pnl)

        # --- Decision Logic ---
        final_pnls = np.array(final_pnls)
        win_probability = np.mean(final_pnls > 0)
        
        avg_profit = np.mean(final_pnls[final_pnls > 0]) if (final_pnls > 0).any() else 0
        avg_loss = np.abs(np.mean(final_pnls[final_pnls < 0])) if (final_pnls < 0).any() else 1e-9
        
        profit_factor = avg_profit / avg_loss

        logger.info(f"Validation results: Win Probability = {win_probability*100:.1f}%, Profit Factor = {profit_factor:.2f}")

        # --- VALIDATION CRITERIA ---
        # The trade is only approved if it has a high probability of winning AND
        # the potential profits significantly outweigh the potential losses.
        if win_probability > 0.60 and profit_factor > 1.5:
            logger.info("✅ Trade VALIDATED. High probability of favorable outcome.")
            return True
        else:
            logger.info("❌ Trade REJECTED. Outcome is too uncertain or risk/reward is poor.")
            return False

    def _simulate_one_path(self, action, price_path, atr):
        """ Simulates a single trade on a given price path with fixed SL/TP. """
        entry_price = price_path[0]
        stop_loss_price = entry_price - (2 * atr) if action > 0 else entry_price + (2 * atr)
        take_profit_price = entry_price + (3 * atr) if action > 0 else entry_price - (3 * atr)

        for price in price_path:
            if action > 0: # Long position
                if price <= stop_loss_price: return stop_loss_price - entry_price
                if price >= take_profit_price: return take_profit_price - entry_price
            else: # Short position (if implemented)
                pass
        
        return price_path[-1] - entry_price