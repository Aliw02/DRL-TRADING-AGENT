# utils/custom_indicators.py (FINAL & COMPREHENSIVE VERSION)

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
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("Index is not DatetimeIndex. Skipping multi-timeframe features.")
        return df
        
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
    # ATR is calculated in the main function, so we just use it here.
    if 'atr' not in df.columns:
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
    """
    logger.info("Engineering candlestick pattern features...")
    op, hi, lo, cl = df['open'], df['high'], df['low'], df['close']
    
    pattern_functions = {
        'CDL2CROWS': ta.CDL2CROWS, 'CDL3BLACKCROWS': ta.CDL3BLACKCROWS,
        'CDLENGULFING': ta.CDLENGULFING, 'CDLHAMMER': ta.CDLHAMMER,
        'CDLHARAMI': ta.CDLHARAMI, 'CDLINVERTEDHAMMER': ta.CDLINVERTEDHAMMER,
        'CDLSHOOTINGSTAR': ta.CDLSHOOTINGSTAR, 'CDLDOJI': ta.CDLDOJI,
    }
    
    for name, func in pattern_functions.items():
        df[name.lower()] = func(op, hi, lo, cl) / 100
        
    return df

def _add_time_features(df: pd.DataFrame):
    """
    Engineers cyclical time-based features for hour and day of the week.
    """
    logger.info("Engineering cyclical time-based features...")
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        day = df.index.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * day / 7)
        df['day_cos'] = np.cos(2 * np.pi * day / 7)
    return df

# --- Main Calculation Function ---

def calculate_all_indicators(df: pd.DataFrame):
    """
    Calculate a comprehensive suite of technical indicators, combining the original
    set with new professional indicators focused on volume and order flow.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.set_index(pd.to_datetime(df['time']), inplace=True)
        except Exception as e:
            logger.error(f"Could not convert index to DatetimeIndex. Error: {e}")
            return df

    logger.info(f"Initial data shape for feature engineering: {df.shape}")
    
    op, hi, lo, cl, vol = df['open'], df['high'], df['low'], df['close'], df['volume']

    # --- Step 1: Calculate Original Base Indicators ---
    logger.info("Calculating original base indicators...")
    df['rsi'] = ta.RSI(cl, timeperiod=14)
    df['adx'] = ta.ADX(hi, lo, cl, timeperiod=14)
    df['macd'], df['macd_signal'], _ = ta.MACD(cl, fastperiod=12, slowperiod=26, signalperiod=9)
    upper_band, middle_band, lower_band = ta.BBANDS(cl, timeperiod=20)
    safe_middle_band = np.where(middle_band == 0, 1e-9, middle_band)
    df['bb_width'] = (upper_band - lower_band) / safe_middle_band
    df['roc'] = ta.ROC(cl, timeperiod=10)
    df['atr'] = ta.ATR(hi, lo, cl, timeperiod=14)

    # --- Step 2: Add NEW Professional Indicators ---
    logger.info("Calculating NEW professional indicators (Volume & Order Flow)...")
    # On-Balance Volume (OBV)
    df['obv'] = ta.OBV(cl, vol)
    # Chaikin Money Flow (CMF)
    ad = ta.AD(hi, lo, cl, vol)
    df['cmf'] = ad.rolling(20).sum() / vol.rolling(20).sum()
    # Vortex Indicator (VI)
    df['vi_plus'] = ta.PLUS_DI(hi, lo, cl, timeperiod=14)
    df['vi_minus'] = ta.MINUS_DI(hi, lo, cl, timeperiod=14)
    
    # --- Step 3: Engineer Original Contextual & Advanced Features ---
    df = _add_multi_timeframe_features(df)
    df = _add_volatility_normalized_features(df)
    df = _add_candlestick_features(df)
    df = _add_time_features(df)

    # --- Step 4: Final Cleanup ---
    logger.info("Cleaning up final dataframe...")
    # The bfill followed by ffill is a robust way to handle NaNs
    processed_df = df.bfill().ffill()
    
    if processed_df.isnull().sum().sum() > 0:
        logger.warning("NaNs still present after cleanup. Filling with 0.")
        processed_df.fillna(0, inplace=True)

    logger.info(f"Feature engineering complete. Final data shape: {processed_df.shape}")
    
    return processed_df
