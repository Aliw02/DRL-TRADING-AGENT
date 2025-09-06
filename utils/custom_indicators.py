# utils/custom_indicators.py (CORRECTED - rsi_x_adx feature added back)
import pandas as pd
import numpy as np
import talib as ta
from utils.logger import get_logger

logger = get_logger(__name__)


def _add_chandelier_exit_signals(df: pd.DataFrame, period=7, multiplier=4.0):
    """Calculates Chandelier Exit and adds binary flip signals."""
    logger.info("Engineering Chandelier Exit flip signals...")

    atr = ta.ATR(df['high'], df['low'], df['close'], timeperiod=period)

    long_stop = df['high'].rolling(period).max() - atr * multiplier
    short_stop = df['low'].rolling(period).min() + atr * multiplier

    # Initialize columns
    df['ce_direction'] = 0
    df['bullish_flip'] = 0
    df['bearish_flip'] = 0

    # Loop to determine direction and flips
    for i in range(1, len(df)):
        # Determine trend direction
        if df['close'].iloc[i] > long_stop.iloc[i-1]:
            df['ce_direction'].iloc[i] = 1
        elif df['close'].iloc[i] < short_stop.iloc[i-1]:
            df['ce_direction'].iloc[i] = -1
        else:
            df['ce_direction'].iloc[i] = df['ce_direction'].iloc[i-1]

        # Detect flips
        if df['ce_direction'].iloc[i] == 1 and df['ce_direction'].iloc[i-1] == -1:
            df['bullish_flip'].iloc[i] = 1 # تلميح: حدث انعكاس صاعد الآن!

        if df['ce_direction'].iloc[i] == -1 and df['ce_direction'].iloc[i-1] == 1:
            df['bearish_flip'].iloc[i] = 1 # تلميح: حدث انعكاس هابط الآن!

    return df

def _add_multi_timeframe_features(df: pd.DataFrame):
    if not isinstance(df.index, pd.DatetimeIndex): return df
    logger.info("Engineering multi-timeframe features...")
    df_m15 = df['close'].resample('15min').ohlc()
    df_h1 = df['close'].resample('1h').ohlc()
    df['rsi_m15'] = ta.RSI(df_m15['close'], timeperiod=14).reindex(df.index, method='ffill')
    df['rsi_h1'] = ta.RSI(df_h1['close'], timeperiod=14).reindex(df.index, method='ffill')
    df['adx_m15'] = ta.ADX(df_m15['high'], df_m15['low'], df_m15['close'], timeperiod=14).reindex(df.index, method='ffill')
    df['adx_h1'] = ta.ADX(df_h1['high'], df_h1['low'], df_h1['close'], timeperiod=14).reindex(df.index, method='ffill')
    return df

def _add_volatility_normalized_features(df: pd.DataFrame):
    logger.info("Engineering volatility-normalized features...")
    if 'atr' not in df.columns:
        df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    safe_atr = df['atr'].replace(0, 1e-9)
    if 'macd' in df.columns: df['macd_norm'] = df['macd'] / safe_atr
    if 'roc' in df.columns: df['roc_norm'] = df['roc'] / safe_atr
    return df

def _add_candlestick_features(df: pd.DataFrame):
    logger.info("Engineering candlestick pattern features...")
    op, hi, lo, cl = df['open'], df['high'], df['low'], df['close']
    patterns = {'CDLENGULFING': ta.CDLENGULFING, 'CDLHAMMER': ta.CDLHAMMER, 'CDLDOJI': ta.CDLDOJI}
    for name, func in patterns.items():
        df[name.lower()] = func(op, hi, lo, cl) / 100
    return df

def _add_time_features(df: pd.DataFrame):
    logger.info("Engineering cyclical time-based features...")
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        df['hour_sin'], df['hour_cos'] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
        day = df.index.dayofweek
        df['day_sin'], df['day_cos'] = np.sin(2*np.pi*day/7), np.cos(2*np.pi*day/7)
    return df

def _add_advanced_statistical_features(df: pd.DataFrame, window=20):
    """Calculates rolling skewness and kurtosis of returns."""
    logger.info("Engineering advanced statistical features (Skew/Kurtosis)...")
    
    # Calculate log returns for better statistical properties
    returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    
    df[f'returns_skew_{window}'] = returns.rolling(window=window).skew()
    df[f'returns_kurt_{window}'] = returns.rolling(window=window).kurt()
    
    return df

def calculate_all_indicators(df: pd.DataFrame):
    if not isinstance(df.index, pd.DatetimeIndex):
        try: df.set_index(pd.to_datetime(df['time']), inplace=True)
        except: return df

    op, hi, lo, cl, vol = df['open'], df['high'], df['low'], df['close'], df['volume']
    
    logger.info("Calculating base indicators...")
    df['rsi'] = ta.RSI(cl)
    df['adx'] = ta.ADX(hi, lo, cl)
    df['macd'], df['macd_signal'], _ = ta.MACD(cl)
    upper, middle, lower = ta.BBANDS(cl)
    df['bb_width'] = (upper - lower) / np.where(middle==0, 1e-9, middle)
    df['roc'] = ta.ROC(cl)
    df['atr'] = ta.ATR(hi, lo, cl)
    df['obv'] = ta.OBV(cl, vol)
    
    # Chaikin Money Flow (CMF)
    ad = ta.AD(hi, lo, cl, vol)
    df['cmf'] = ad.rolling(20).sum() / vol.rolling(20).sum()

    df['vi_plus'] = ta.PLUS_DI(hi, lo, cl)
    df['vi_minus'] = ta.MINUS_DI(hi, lo, cl)
    
    df = _add_multi_timeframe_features(df)
    df = _add_volatility_normalized_features(df)
    
    # --- CRITICAL FIX: Calculate the missing 'rsi_x_adx' feature ---
    logger.info("Engineering feature interactions...")
    df['rsi_x_adx'] = (df['rsi'] / 100) * (df['adx'] / 100)
    
    df = _add_candlestick_features(df)
    df = _add_time_features(df)
    df = _add_advanced_statistical_features(df) 
    df = _add_chandelier_exit_signals(df) 

    logger.info("Cleaning up final dataframe...")
    return df.bfill().ffill().fillna(0)