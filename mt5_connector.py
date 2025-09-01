# mt5_connector.py
# mt5_connector.py (No major changes, just ensure it can handle larger requests)
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time

def connect_mt5(login, password, server):
    """
    يتصل بمنصة MT5 ويسجل الدخول باستخدام بيانات الاعتماد.
    """
    if not mt5.initialize():
        print("initialize() failed, error: ", mt5.last_error())
        return False
        
    authorized = mt5.login(login, password, server)
    if not authorized:
        print("Login failed, error: ", mt5.last_error())
        mt5.shutdown()
        return False

    print("Connected and logged in successfully!")
    return True

def get_mt5_data(symbol, timeframe, num_bars):
    """
    يحصل على البيانات التاريخية من MT5 ويقوم بتوحيد اسم عمود الحجم.
    """
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        if rates is None or len(rates) == 0:
            print("Failed to get data.")
            return None
        
        df = pd.DataFrame(rates)
        
        # --- هذا هو السطر الذي يحل المشكلة ---
        # MT5 provides 'tick_volume'. We rename it to 'volume' to match the
        # column name used throughout the entire training pipeline.
        if 'tick_volume' in df.columns:
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        else:
            # As a fallback, create a volume column of zeros if not present
            df['volume'] = 0
            
        # The 'time' column is essential for resampling
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('timestamp', inplace=True)
        
        # We only need the OHLCV columns for our model
        return df[['open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        print(f"Error occurred while fetching data: {e}")
        return None
        
def get_current_data(symbol):
    """
    يحصل على أحدث سعر إغلاق.
    """
    try:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
        if rates is None or len(rates) == 0:
            print("فشل في الحصول على السعر الحالي.")
            return None
        return rates[0]['close']
    except Exception as e:
        print(f"حدث خطأ أثناء جلب السعر الحالي: {e}")
        return None