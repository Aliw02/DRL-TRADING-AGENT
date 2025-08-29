# Technical indicators implementation
import pandas as pd
import numpy as np
from numba import jit
from utils.logger import get_logger

logger = get_logger(__name__)

import pandas as pd
import numpy as np
import talib as ta

# --- Helper functions for Custom Calculations (prefixed with _) ---

def _calculate_chandelier_exit(high, low, close, period=7, multiplier=4):
    """Custom calculation for Chandelier Exit."""
    atr = ta.ATR(high, low, close, timeperiod=period)
    
    long_stop = high.rolling(window=period).max() - atr * multiplier
    short_stop = low.rolling(window=period).min() + atr * multiplier
    
    # Use shifted values for comparing with current close
    direction = np.where(close > long_stop.shift(1), 1, 
                         np.where(close < short_stop.shift(1), -1, 0))
    direction = pd.Series(direction, index=close.index).ffill().fillna(0)
    
    bullish_flip = ((direction == 1) & (direction.shift(1) == -1)).astype(int)
    bearish_flip = ((direction == -1) & (direction.shift(1) == 1)).astype(int)
    
    dist_to_stop = np.where(direction == 1, close - long_stop, short_stop - close)
    
    return direction, bullish_flip, bearish_flip, dist_to_stop, long_stop, short_stop

def _add_fibonacci_levels(df: pd.DataFrame, window=100):
    """Calculates objective Fibonacci Retracement levels."""
    rolling_high = df['high'].rolling(window=window, min_periods=window).max()
    rolling_low = df['low'].rolling(window=window, min_periods=window).min()

    diff = rolling_high - rolling_low
    df['dist_to_fib_0.382'] = abs(df['close'] - (rolling_high - diff * 0.382))
    df['dist_to_fib_0.618'] = abs(df['close'] - (rolling_high - diff * 0.618))
    
    return df

def _add_advanced_interaction_features(df: pd.DataFrame):
    """Calculates interaction and market regime features."""
    df['rsi_x_adx'] = (df['rsi'] / 100) * (df['adx'] / 100) * 100
    bbw_mean = df['bb_width'].rolling(window=50).mean()
    df['market_regime'] = np.where(df['bb_width'] > bbw_mean, 1, 0)

    return df

def _add_contextual_features(df: pd.DataFrame):
    """Adds features based on price action, time, and momentum dynamics."""
    body_size = abs(df['close'] - df['open'])
    candle_range = df['high'] - df['low']
    df['body_to_wick_ratio'] = body_size / candle_range.replace(0, 0.0001)

    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    df['rsi_change'] = df['rsi'].diff()
    df['dist_from_ma_slow'] = (df['close'] - df['ma_slow']) / df['ma_slow']

    return df

# --- Main Calculation Function ---

def calculate_all_indicators(df: pd.DataFrame):
    """
    Calculate all technical indicators and advanced features using TA-Lib for speed and accuracy.
    This is the main function to be called from the data transformation step.
    """
    # Ensure DataFrame index is a datetime object for time-based features
    if not isinstance(df.index, pd.DatetimeIndex):
        # Attempt to convert the index, assuming it's a valid datetime format
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            print(f"Warning: Could not convert index to DatetimeIndex. Time-based features will be skipped. Error: {e}")


    # --- Step 1: Calculate Base Indicators using TA-Lib ---
    # TA-Lib requires numpy arrays as input (float64)
    # Ensure inputs are double precision numpy arrays to avoid TA-Lib errors
    high_prices = np.asarray(df['high'].values, dtype='double')
    low_prices = np.asarray(df['low'].values, dtype='double')
    close_prices = np.asarray(df['close'].values, dtype='double')
    
    # Standard Indicators
    df['rsi'] = ta.RSI(close_prices, timeperiod=14)
    df['adx'] = ta.ADX(high_prices, low_prices, close_prices, timeperiod=14)
    df['plus_di'] = ta.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
    df['minus_di'] = ta.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
    df['ma_fast'] = ta.SMA(close_prices, timeperiod=10)
    df['ma_slow'] = ta.SMA(close_prices, timeperiod=50)
    
    macd, macd_signal, _ = ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    
    upper_band, middle_band, lower_band = ta.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    # Avoid division by zero for bb_width
    middle_band[middle_band == 0] = 1e-6 # a small number to prevent error
    df['bb_width'] = (upper_band - lower_band) / middle_band
    
    df['roc'] = ta.ROC(close_prices, timeperiod=10)
    df['volatility'] = ta.STDDEV(close_prices, timeperiod=20)
    if 'volume' in df.columns:
        # Ensure volume array is double precision for TA-Lib
        volume_values = np.asarray(df['volume'].values, dtype='double')
        df['volume_ma'] = ta.SMA(volume_values, timeperiod=20)
        # Avoid division by zero for volume_ratio
        volume_ma_safe = df['volume_ma'].replace(0, 1e-6)
        df['volume_ratio'] = df['volume'] / volume_ma_safe
        df['volume_ratio'] = df['volume'] / volume_ma_safe

    # --- Step 2: Calculate Custom Indicators & Features ---
    # Chandelier Exit is custom and remains as a manual calculation
   
    # --- HERE IS THE FIX ---
    direction, bullish_flip, bearish_flip, dist_to_stop, long_stop, short_stop = _calculate_chandelier_exit(df['high'], df['low'], df['close'])
    
    df['direction'] = direction
    df['bullish_flip'] = bullish_flip
    df['bearish_flip'] = bearish_flip
    df['dist_to_stop'] = dist_to_stop
    df['ce_ground'] = long_stop  
    df['ce_roof'] = short_stop   


    # --- Step 3: Calculate Advanced Level-Based and Contextual Features ---
    df = _add_fibonacci_levels(df, window=100)
    df = _add_advanced_interaction_features(df)
    df = _add_contextual_features(df)

    # --- Step 4: Final Cleanup ---
    # Using forward-fill is often safer for time-series to avoid lookahead bias.
    # --- الكود الجديد والصحيح الذي يزيل التحذير ---
    df.ffill(inplace=True)
    df.bfill(inplace=True) # Back-fill any NaNs at the beginning
    
    print("All indicators and features calculated successfully using TA-Lib.")
    return df
