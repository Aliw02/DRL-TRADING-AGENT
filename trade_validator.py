# trade_validator.py (NEW MODULE FOR TRADE SIMULATION)
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

class TradeValidator:
    def __init__(self, agent_model, preprocessor, historical_data: pd.DataFrame):
        """
        Initializes the validator with the trained agent, preprocessor, and historical data.
        
        Args:
            agent_model: The trained SAC model.
            preprocessor: The LiveDataPreprocessor instance.
            historical_data (pd.DataFrame): A recent chunk of historical data for simulation.
        """
        self.model = agent_model
        self.preprocessor = preprocessor
        self.historical_data = historical_data
        self.sequence_length = self.model.policy.observation_space.shape[0]

    def _generate_future_scenarios(self, last_known_price, num_scenarios=50, num_steps=15):
        """
        Generates multiple possible future price paths using historical volatility.
        This is a simplified Monte Carlo simulation.
        """
        # Calculate historical volatility (standard deviation of log returns)
        log_returns = np.log(self.historical_data['close'] / self.historical_data['close'].shift(1))
        volatility = log_returns.std()

        # Generate future paths
        scenarios = np.zeros((num_scenarios, num_steps))
        for i in range(num_scenarios):
            # Create random shocks based on historical volatility
            shocks = np.random.normal(0, volatility, num_steps)
            path = last_known_price * np.exp(np.cumsum(shocks))
            scenarios[i, :] = path
        
        return scenarios

    def validate_trade(self, current_observation, initial_action):
        """
        Runs a micro-backtest on the proposed trade to check its viability.

        Returns:
            bool: True if the trade is validated, False otherwise.
        """
        logger.info(f"🔬 Validating proposed trade action: {initial_action:.2f}...")
        
        last_real_price = self.historical_data['close'].iloc[-1]
        future_scenarios = self._generate_future_scenarios(last_real_price)
        
        final_pnls = []

        for scenario in future_scenarios:
            # Simulate the trade over this specific price path
            hypothetical_pnl = self._simulate_one_path(current_observation, initial_action, scenario)
            final_pnls.append(hypothetical_pnl)

        # --- Decision Logic ---
        final_pnls = np.array(final_pnls)
        
        # Calculate the probability of the trade being profitable
        win_probability = np.mean(final_pnls > 0)
        
        # Calculate the average profit vs. average loss
        avg_profit = np.mean(final_pnls[final_pnls > 0]) if (final_pnls > 0).any() else 0
        avg_loss = np.abs(np.mean(final_pnls[final_pnls < 0])) if (final_pnls < 0).any() else 1
        
        profit_factor = avg_profit / avg_loss if avg_loss > 0 else avg_profit

        logger.info(f"Validation results: Win Probability = {win_probability*100:.1f}%, Profit Factor = {profit_factor:.2f}")

        # VALIDATION CRITERIA: We only approve the trade if it has a high chance of winning
        # AND the potential profits outweigh the potential losses.
        if win_probability > 0.60 and profit_factor > 1.5:
            logger.info("✅ Trade VALIDATED. High probability of success.")
            return True
        else:
            logger.info("❌ Trade REJECTED. Outcome is too uncertain.")
            return False

    def _simulate_one_path(self, observation, action, price_path):
        """ Simulates a single trade from entry to exit on a given price path. """
        entry_price = price_path[0]
        current_position = action # e.g., 0.8 for BUY, -0.7 for SELL
        
        # Simulate the next few steps
        for i in range(1, len(price_path)):
            current_price = price_path[i]
            
            # Here we would need to update the observation with the new price
            # For simplicity, we assume the model's exit logic is quick and based on price movement
            # A more complex simulation would update all indicators and re-run the preprocessor
            
            # Simple Exit Logic: If the model's action flips, we exit
            # (This is a simplification; a full implementation is much more complex)
            # For now, let's assume exit is based on a simple stop-loss/take-profit
            pnl = (current_price - entry_price) * current_position
            
            # Simple risk management: exit if loss > 2*ATR or profit > 3*ATR
            atr = self.historical_data['atr'].iloc[-1]
            if pnl < -2 * atr or pnl > 3 * atr:
                return pnl # Exit trade

        # If not exited, return the final P&L
        final_pnl = (price_path[-1] - entry_price) * current_position
        return final_pnl