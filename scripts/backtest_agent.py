# backtester.py
import pandas as pd
import joblib
import numpy as np
import os
import sys

# --- Add project path to access other modules ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.custom_indicators import calculate_all_indicators
from sb3_contrib import RecurrentPPO
from utils.data_transformation import DataTransformer
from config.init import config
from envs.trading_env import TradingEnv

# --- Final file paths ---
FINAL_MODEL_PATH = "results/final_model_for_live/finetuned_model.zip"
FINAL_SCALER_PATH = "results/final_model_for_live/final_robust_scaler.joblib"
# Ensure this file exists
TIMEFRAME = 15
TEST_DATA_PATH = f"data/XAUUSDM{TIMEFRAME}-TEST_UNSEEN.csv"

# --- NEW BACKTESTER SETTINGS ---
USE_DYNAMIC_TP = False

# --- NEW FILTER TOGGLES ---
USE_RSI_FILTER = False
USE_ADX_FILTER = False
USE_CE_FLIP_FILTER = False

COMMISSION_PCT = config.get('environment.commission_pct', 0.0005)
SLIPPAGE_PCT = config.get('environment.slippage_pct', 0.0002)


def load_tickstory_data(file_path):
    """
    Loads and prepares Tickstory data.
    Ensures a dedicated 'time' column is present.
    """
    column_names = ['date', 'time', 'open', 'high', 'low', 'close', 'tick_volume', 'volume', 'spread']
    df = pd.read_csv(file_path, sep=',', header=None, names=column_names, dtype={'date': str, 'time': str})
    
    # Combine date and time into a single datetime column
    df['time'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y%m%d %H:%M:%S', errors='coerce')
    df.drop(columns=['date'], inplace=True)
    
    # Ensure a single 'volume' column for downstream processing
    if 'volume' not in df.columns and 'tick_volume' in df.columns:
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    elif 'tick_volume' in df.columns and 'volume' in df.columns:
        df.drop(columns=['tick_volume'], inplace=True)
    
    df = df.drop(columns=['spread'], errors='ignore')
    return df


def run_backtest():
    """
    Runs a full trading simulation on new data.
    """
    print("===== Starting Backtesting Process =====")
    
    try:
        model = RecurrentPPO.load(FINAL_MODEL_PATH)
        scaler = joblib.load(FINAL_SCALER_PATH)
        print("Model and Scaler loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Model or Scaler files not found. {e}")
        return

    test_df = load_tickstory_data(file_path=TEST_DATA_PATH)
    test_df = calculate_all_indicators(test_df) # Calculate all indicators on the test data

    # --- THE FIX: SEPARATING DATA FOR MODEL AND FOR LOGIC ---
    # The list of features the model was trained on
    model_feature_cols = ['direction', 'bullish_flip', 'bearish_flip', 'dist_to_stop', 'rsi']

    # DataFrame for the model's observations (only the features it was trained on)
    obs_df_scaled = pd.DataFrame(scaler.transform(test_df[model_feature_cols]), columns=model_feature_cols)

    # DataFrame for the backtesting logic (all other columns)
    logic_df = test_df.copy()
    
    # Add 'close' and 'time' columns to obs_df_scaled for TradingEnv
    obs_df_scaled['close'] = logic_df['close'].values
    obs_df_scaled['time'] = logic_df['time'].values

    print(f"Loaded {len(obs_df_scaled)} data points for testing.")

    backtest_env = TradingEnv(obs_df_scaled, initial_balance=1000)
    obs, _ = backtest_env.reset()
    
    states = None
    done = False
    
    equity_curve = []
    trade_history = []
    initial_balance = backtest_env.initial_balance

    open_position_type = None
    open_position_entry_time = None
    open_position_entry_price = 0
    
    print("Starting simulation...")
    while not done:
        current_step = backtest_env.current_step
        
        action, states = model.predict(obs, state=states, deterministic=True)
        current_time = logic_df.iloc[current_step]['time']
        current_price = logic_df.iloc[current_step]['close']
        
        # --- NEW ENTRY LOGIC (Filtering the DRL model's action) ---
        can_trade = True
        # Check if enough data for indicators
        if current_step >= 50:
            current_adx = logic_df.iloc[current_step]['adx']
            current_rsi = logic_df.iloc[current_step]['rsi']
            current_bullish_flip = logic_df.iloc[current_step]['bullish_flip']
            current_bearish_flip = logic_df.iloc[current_step]['bearish_flip']

            # --- APPLY FILTERS BASED ON TOGGLES ---
            if USE_ADX_FILTER and current_adx < 25:
                can_trade = False
            
            if USE_RSI_FILTER:
                if (action == 1 and current_rsi < 50) or (action == 2 and current_rsi > 50):
                    can_trade = False

            if USE_CE_FLIP_FILTER:
                if (action == 1 and current_bullish_flip != 1) or (action == 2 and current_bearish_flip != 1):
                    can_trade = False
        else:
            can_trade = False # Not enough data for indicators at the start

        # --- TRADE MANAGEMENT LOGIC ---
        if open_position_type is None and can_trade:
            if action == 1:
                open_position_type = 'buy'
                open_position_entry_time = current_time
                open_position_entry_price = current_price * (1 + SLIPPAGE_PCT)
            elif action == 2:
                open_position_type = 'sell'
                open_position_entry_time = current_time
                open_position_entry_price = current_price * (1 - SLIPPAGE_PCT)
        
        # --- CLOSE LOGIC (Dynamic TP/SL or manual close) ---
        if open_position_type:
            close_trade = False
            
            # Get dynamic indicators for exit
            current_bullish_flip = logic_df.iloc[current_step]['bullish_flip']
            current_bearish_flip = logic_df.iloc[current_step]['bearish_flip']
            ce_ground_line = logic_df.iloc[current_step]['ce_ground']
            ce_roof_line = logic_df.iloc[current_step]['ce_roof']

            if USE_DYNAMIC_TP:
                if open_position_type == 'buy':
                    if current_price < ce_ground_line: # Hit dynamic SL
                        close_trade = True
                    elif current_bearish_flip == 1: # Hit dynamic TP
                        close_trade = True
                elif open_position_type == 'sell':
                    if current_price > ce_roof_line: # Hit dynamic SL
                        close_trade = True
                    elif current_bullish_flip == 1: # Hit dynamic TP
                        close_trade = True
            else: # Manual close logic (e.g., based on DRL action)
                if open_position_type == 'buy' and action == 2:
                    close_trade = True
                elif open_position_type == 'sell' and action == 1:
                    close_trade = True
            
            if close_trade:
                if open_position_type == 'buy':
                    profit_loss = (current_price - open_position_entry_price) * (1 - COMMISSION_PCT)
                else: # 'sell'
                    profit_loss = (open_position_entry_price - current_price) * (1 - COMMISSION_PCT)
                
                trade_history.append({
                    'entry_time': open_position_entry_time,
                    'exit_time': current_time,
                    'profit_loss': profit_loss,
                    'type': 'buy_closed' if open_position_type == 'buy' else 'sell_closed'
                })
                open_position_type = None
                open_position_entry_time = None
                open_position_entry_price = 0
                
        obs, reward, done, _, info = backtest_env.step(action)
        
        current_equity = initial_balance + sum(t['profit_loss'] for t in trade_history)
        
        equity_curve.append({
            'timestamp': current_time,
            'equity': current_equity
        })
        
    # Final PnL for any open trade
    if open_position_type:
        current_price = logic_df.iloc[backtest_env.current_step - 1]['close']
        if open_position_type == 'buy':
            profit_loss = (current_price - open_position_entry_price) * (1 - COMMISSION_PCT)
        else:
            profit_loss = (open_position_entry_price - current_price) * (1 - COMMISSION_PCT)
            
        current_equity += profit_loss
        
        trade_history.append({
            'entry_time': open_position_entry_time,
            'exit_time': current_time,
            'profit_loss': profit_loss,
            'type': 'buy_closed' if open_position_type == 'buy' else 'sell_closed'
        })
        
        equity_curve.append({
            'timestamp': current_time,
            'equity': current_equity
        })
        
    # 4. Save results
    equity_df = pd.DataFrame(equity_curve)
    trade_df = pd.DataFrame(trade_history)

    equity_df.to_csv("results/backtest_equity.csv", index=False)
    trade_df.to_csv("results/backtest_trades.csv", index=False)
    
    print("===== Backtesting Completed =====")
    final_equity = initial_balance + trade_df['profit_loss'].sum()
    print(f"Final Equity: {final_equity:.2f}")
    print(f"Net Profit: {trade_df['profit_loss'].sum():.2f}")
    print(f"Total Trades: {len(trade_df)}")
    
    return equity_df, trade_df

if __name__ == "__main__":
    run_backtest()