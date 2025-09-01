# main_bot.py (CRITICAL UPDATE: Now operates on a 15-minute cycle)
import os
import time
import pandas as pd
import MetaTrader5 as mt5
import sys

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.custom_indicators import calculate_all_indicators
from mt5_connector import connect_mt5, get_mt5_data
from data_preprocessor import LiveDataPreprocessor # We will use the class-based preprocessor
from trading_agent import load_trading_model, get_action_from_model
from trade_executor import execute_trade
from config.init import config

# --- Bot Settings ---
MT5_LOGIN = 779654
MT5_PASSWORD = "l%6CRR)v"
MT5_SERVER = "Inzo-Demo"
SYMBOL = "XAUUSD"
LOT_SIZE = 0.01

# --- Timeframe Settings ---
LIVE_TIMEFRAME = mt5.TIMEFRAME_M1  # We fetch 1-minute data
MODEL_TIMEFRAME = '15min'         # But the model operates on 15-minute data
SEQUENCE_LENGTH = config.get('environment.sequence_length', 150) # From config_sac.yaml

def main():
    if not connect_mt5(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        return

    try:
        preprocessor = LiveDataPreprocessor()
        model = load_trading_model()
        if model is None: return
    except Exception:
        return

    print("🚀 Starting trading bot on a 15-minute cycle...")

    while True:
        # --- 1. Fetch Sufficient 1-Minute Data ---
        # We need enough 1-min bars to create at least SEQUENCE_LENGTH + (some buffer for indicators) of 15-min bars.
        # Calculation: (150 sequences + 50 for indicator stability) * 15 minutes/sequence = 3000 bars
        bars_to_fetch = ((SEQUENCE_LENGTH + 50) * 15) + 1000

        df_m1 = get_mt5_data(SYMBOL, LIVE_TIMEFRAME, bars_to_fetch)
        
        if df_m1 is None or df_m1.empty:
            print("Could not fetch 1-minute data, waiting...")
            time.sleep(60) # Wait 1 minute before retrying
            continue
        
        print(f"Fetched {len(df_m1)} 1-minute bars.")

        # --- 2. Resample Data to 15-Minute Timeframe ---
        # This is the crucial step to match the training data format
        try:
            aggregation_rules = {
                'open': 'first', 'high': 'max',
                'low': 'min', 'close': 'last', 'volume': 'sum'
            }
            df_m15 = df_m1.resample(MODEL_TIMEFRAME).agg(aggregation_rules).dropna()
            print(f"Resampled to {len(df_m15)} 15-minute bars.")
        except Exception as e:
            print(f"Error during resampling: {e}")
            time.sleep(60)
            continue

        # --- 3. Calculate Indicators on the 15-Minute Data ---
        df_with_indicators = calculate_all_indicators(df_m15)

        # --- 4. Preprocess Data for the Model ---
        # The preprocessor now receives the correct 15-minute data
        final_observation = preprocessor.preprocess(df_with_indicators)
        
        if final_observation is None:
            print("Data preprocessing failed, waiting for next cycle.")
            # Wait for the next 15-minute candle
            time.sleep(15 * 60)
            continue
            
        # --- 5. Get Trading Action ---
        action = get_action_from_model(model, final_observation)
        if action is None:
            print("Failed to get action from model, waiting for next cycle.")
            time.sleep(15 * 60)
            continue
        
        # --- 6. Execute Trade ---
        # The SAC model outputs a continuous value. We can interpret it as:
        # > 0.3 = Strong Buy Signal (e.g., action 1)
        # < -0.3 = Strong Sell Signal (e.g., action 2)
        # otherwise = Hold (e.g., action 0)
        
        # This part requires you to adapt `execute_trade` or interpret the action here.
        # Let's create a simple interpretation:
        action_value = action[0][0]
        print(f"Model raw action: {action_value:.2f}")

        # The `execute_trade` function in your project expects discrete actions (0,1,2)
        # Let's convert the continuous SAC action to discrete
        # discrete_action = 0 # Hold
        # if action_value > 0.5: # Threshold for buy
        #     discrete_action = 1 # Buy
        # elif action_value < -0.5: # Threshold for sell
        #     discrete_action = 2 # Sell

        execute_trade(SYMBOL, action_value, LOT_SIZE)

        # --- 7. Wait for the Next 15-Minute Candle ---
        print(f"Decision made. Waiting for the next 15-minute candle...")
        time.sleep(15 * 60)
        
if __name__ == "__main__":
    main()