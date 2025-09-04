# main_bot.py (FINAL, UNCERTAINTY-AWARE VERSION)
import os
import sys
import time
import pandas as pd
import MetaTrader5 as mt5

# --- Add project path to allow imports from other directories ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.custom_indicators import calculate_all_indicators
from mt5_connector import connect_mt5, get_mt5_data
from data_preprocessor import LiveDataPreprocessor
from trading_agent import load_trading_model, get_action_and_uncertainty
from trade_executor import execute_trade
from config.init import config
from utils.logger import setup_logging, get_logger

# --- Setup Logging ---
setup_logging()
logger = get_logger(__name__)

# --- Bot Settings ---
MT5_LOGIN = 779654
MT5_PASSWORD = "l%6CRR)v"
MT5_SERVER = "Inzo-Demo"
SYMBOL = "XAUUSD"
LOT_SIZE = 0.01

# --- Timeframe Settings ---
LIVE_TIMEFRAME = mt5.TIMEFRAME_M1
MODEL_TIMEFRAME = '15min'
SEQUENCE_LENGTH = config.get('environment.sequence_length', 150)

# --- UNCERTAINTY THRESHOLD ---
# We will only trade if the model's action uncertainty (std dev) is BELOW this value.
# This is a hyperparameter you can tune. A lower value makes the bot more cautious.
UNCERTAINTY_THRESHOLD = 0.25

def main():
    """
    Main function to run the uncertainty-aware trading bot.
    """
    if not connect_mt5(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        return

    try:
        logger.info("Initializing bot components...")
        preprocessor = LiveDataPreprocessor()
        model = load_trading_model()
        if model is None:
            logger.error("Failed to load the trading model. Exiting.")
            return
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        return

    logger.info(f"🚀 Starting Uncertainty-Aware trading bot.")
    logger.info(f"Symbol: {SYMBOL}, Model Timeframe: {MODEL_TIMEFRAME}, Uncertainty Threshold: {UNCERTAINTY_THRESHOLD}")

    while True:
        try:
            # --- 1. Fetch and Prepare Data ---
            logger.info("Fetching latest market data...")
            # Fetch enough 1-min bars to create a sufficient history of 15-min bars
            bars_to_fetch = ((SEQUENCE_LENGTH + 100) * 15)
            df_m1 = get_mt5_data(SYMBOL, LIVE_TIMEFRAME, bars_to_fetch)

            if df_m1 is None or df_m1.empty:
                logger.warning("Could not fetch 1-minute data. Waiting for 1 minute.")
                time.sleep(60)
                continue
            
            # Resample to the model's timeframe
            aggregation_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            df_m15 = df_m1.resample(MODEL_TIMEFRAME).agg(aggregation_rules).dropna()
            
            # Calculate indicators and preprocess for the model
            df_with_indicators = calculate_all_indicators(df_m15)
            final_observation = preprocessor.preprocess(df_with_indicators)
            
            if final_observation is None:
                logger.warning("Data preprocessing failed. Waiting for the next 15-minute candle.")
                time.sleep(15 * 60)
                continue
            
            # --- 2. Get Action and Uncertainty from Model ---
            action, uncertainty = get_action_and_uncertainty(model, final_observation)
            
            if action is None:
                logger.error("Failed to get action from model. Waiting for the next cycle.")
                time.sleep(15 * 60)
                continue
            
            action_value = action[0][0]
            uncertainty_value = uncertainty[0][0]
            
            logger.info(f"Model Raw Output -> Action: {action_value:.3f} | Uncertainty: {uncertainty_value:.3f}")

            # --- 3. Make a Decision Based on Confidence ---
            final_decision_value = 0.0 # Default action is to HOLD

            if uncertainty_value > UNCERTAINTY_THRESHOLD:
                logger.warning(f"Model is UNCERTAIN (Uncertainty {uncertainty_value:.3f} > {UNCERTAINTY_THRESHOLD}). Overriding to HOLD.")
            else:
                logger.info(f"Model is CONFIDENT (Uncertainty {uncertainty_value:.3f} <= {UNCERTAINTY_THRESHOLD}). Proceeding with action.")
                # We still need a strong signal to act
                if abs(action_value) > 0.5:
                     final_decision_value = action_value
                else:
                    logger.info(f"Signal is too weak (|{action_value:.3f}| <= 0.5). Holding position.")


            # --- 4. Execute the Final Decision ---
            execute_trade(SYMBOL, final_decision_value, LOT_SIZE)

            # --- 5. Wait for the Next Cycle ---
            logger.info("Decision complete. Waiting for the next 15-minute candle...")
            time.sleep(15 * 60)

        except Exception as e:
            logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            time.sleep(60) # Wait a minute before retrying after a major error
        
if __name__ == "__main__":
    main()