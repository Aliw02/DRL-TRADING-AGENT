# main_bot.py (Updated)
import os
import time
import MetaTrader5 as mt5
from utils.custom_indicators import calculate_all_indicators
from mt5_connector import connect_mt5, get_mt5_data
from data_preprocessor import load_scaler, preprocess_data
from trading_agent import load_trading_model, get_action_from_model
from trade_executor import execute_trade, get_open_position # import the new function
from config.init import config

# --- إعدادات البوت، قم بتغيير هذه القيم ---
"""
Account: 779654
Password: l%6CRR)v
Server: Inzo-Demo
"""
MT5_LOGIN = 779654
MT5_PASSWORD = "l%6CRR)v"
MT5_SERVER = "Inzo-Demo"
SYMBOL = "XAUUSD"
TIMEFRAME = 1
NUM_BARS_REQUIRED = config.get('environment.sequence_length', 40)
LOT_SIZE = 0.01

def main():
    if not connect_mt5(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        return

    scaler = load_scaler()
    if scaler is None:
        return

    model = load_trading_model()
    if model is None:
        return

    print("Starting trading process...")

    # تهيئة حالة النموذج الأولية
    states = None

    while True:
        # 1. الحصول على البيانات
        df = get_mt5_data(SYMBOL, eval(f"mt5.TIMEFRAME_M{TIMEFRAME}"), NUM_BARS_REQUIRED + 1)
        if df is None:
            time.sleep(TIMEFRAME * 60)
            continue
        
        
        # 2. الحصول على الموقف الحالي للتداول
        current_position = get_open_position(SYMBOL)
        
        df = calculate_all_indicators(df)
        # 3. تجهيز البيانات للنموذج مع الموقف الحالي
        processed_data = preprocess_data(df, scaler, current_position)
        if processed_data is None:
            time.sleep(TIMEFRAME * 60)
            continue
            
        # 4. الحصول على قرار التداول وتحديث الحالة
        action, states = get_action_from_model(model, processed_data, states)
        if action is None:
            time.sleep(TIMEFRAME * 60)
            continue
        
        # 5. تنفيذ القرار
        execute_trade(SYMBOL, action[0], LOT_SIZE)

        print(f"Waiting for {TIMEFRAME} minutes...")
        time.sleep(TIMEFRAME * 60)
        
if __name__ == "__main__":
    main()