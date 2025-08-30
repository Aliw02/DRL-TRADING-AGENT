# utils/custom_indicators.py (FINAL PROFESSIONAL VERSION)

import pandas as pd
import numpy as np
import talib as ta
from utils.logger import get_logger

logger = get_logger(__name__)

# --- Helper functions for advanced features ---

def _add_multi_timeframe_features(df: pd.DataFrame):
    """
    Engineers features from higher timeframes (M15, H1) to provide broader context.
    """
    logger.info("Engineering multi-timeframe features (M15, H1)...")
    df_m15 = df['close'].resample('15min').ohlc()
    df_h1 = df['close'].resample('1h').ohlc()
    
    df['rsi_m15'] = ta.RSI(df_m15['close'], timeperiod=14).reindex(df.index, method='ffill')
    df['rsi_h1'] = ta.RSI(df_h1['close'], timeperiod=14).reindex(df.index, method='ffill')
    df['adx_m15'] = ta.ADX(df_m15['high'], df_m15['low'], df_m15['close'], timeperiod=14).reindex(df.index, method='ffill')
    df['adx_h1'] = ta.ADX(df_h1['high'], df_h1['low'], df_h1['close'], timeperiod=14).reindex(df.index, method='ffill')
    return df

def _add_volatility_normalized_features(df: pd.DataFrame):
    """
    Normalizes non-bounded indicators by volatility (ATR).
    """
    logger.info("Engineering volatility-normalized features...")
    df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    safe_atr = df['atr'].replace(0, 1e-9)
    if 'macd' in df.columns:
        df['macd_norm'] = df['macd'] / safe_atr
    if 'roc' in df.columns:
        df['roc_norm'] = df['roc'] / safe_atr
    return df

def _add_candlestick_features(df: pd.DataFrame):
    """
    Engineers features based on common candlestick patterns.
    The output is binary (100 for pattern found, 0 otherwise).
    """
    logger.info("Engineering candlestick pattern features...")
    op, hi, lo, cl = df['open'], df['high'], df['low'], df['close']
    
    # List of TA-Lib candlestick pattern functions to run
    pattern_functions = {
        'CDL2CROWS': ta.CDL2CROWS,
        'CDL3BLACKCROWS': ta.CDL3BLACKCROWS,
        'CDLENGULFING': ta.CDLENGULFING,
        'CDLHAMMER': ta.CDLHAMMER,
        'CDLHARAMI': ta.CDLHARAMI,
        'CDLINVERTEDHAMMER': ta.CDLINVERTEDHAMMER,
        'CDLSHOOTINGSTAR': ta.CDLSHOOTINGSTAR,
        'CDLDOJI': ta.CDLDOJI,
    }
    
    for name, func in pattern_functions.items():
        # Divide by 100 to get a simple 1 (bullish), -1 (bearish), or 0 signal
        df[name] = func(op, hi, lo, cl) / 100
        
    return df

def _add_time_features(df: pd.DataFrame):
    """
    Engineers cyclical time-based features for hour and day of the week.
    This helps the model understand weekly and daily seasonality.
    """
    logger.info("Engineering cyclical time-based features...")
    # Make sure index is datetime
    if isinstance(df.index, pd.DatetimeIndex):
        # Hour features
        hour = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week features
        day = df.index.dayofweek # Monday=0, Sunday=6
        df['day_sin'] = np.sin(2 * np.pi * day / 7)
        df['day_cos'] = np.cos(2 * np.pi * day / 7)
    return df


# --- Main Calculation Function (Updated) ---

def calculate_all_indicators(df: pd.DataFrame):
    """
    Calculate all technical indicators and advanced features.
    This is the main function to be called from the data transformation step.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.set_index(pd.to_datetime(df['time']), inplace=True)
        except Exception as e:
            logger.error(f"Could not convert index to DatetimeIndex. Time-based features will be skipped. Error: {e}")
            return df

    # --- Step 1: Calculate Base Indicators using TA-Lib ---
    logger.info("Calculating base TA-Lib indicators...")
    logger.info(f"Data Shape before feature engineering: {df.shape}")
    high, low, close = df['high'].values, df['low'].values, df['close'].values
    
    df['rsi'] = ta.RSI(close, timeperiod=14)
    df['adx'] = ta.ADX(high, low, close, timeperiod=14)
    df['macd'], df['macd_signal'], _ = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    
    # --- FIX: Correctly unpack and calculate Bollinger Bands ---
    upper_band, middle_band, lower_band = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    # Calculate width relative to the middle band to avoid division by zero
    safe_middle_band = np.where(middle_band == 0, 1e-9, middle_band)
    df['bb_width'] = (upper_band - lower_band) / safe_middle_band
    
    df['roc'] = ta.ROC(close, timeperiod=10)
    
    # --- Step 2: Engineer Advanced and Contextual Features ---
    df = _add_multi_timeframe_features(df)
    df = _add_volatility_normalized_features(df)
    df = _add_candlestick_features(df)
    df = _add_time_features(df)

    # --- Step 3: Final Cleanup ---
    logger.info("Cleaning up NaN values from feature engineering...")
    # Drop columns that are intermediate calculations and not features
    df.drop(columns=['atr'], inplace=True, errors='ignore')
    df.ffill(inplace=True)
    df.bfill(inplace=True) 

    logger.info(f"Data Shape after feature engineering: {df.shape}")
    logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
    return df