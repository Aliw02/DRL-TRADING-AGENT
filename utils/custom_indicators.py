# utils/custom_indicators.py (UPGRADED WITH NUMBA FOR MAXIMUM PERFORMANCE)
import pandas as pd
import numpy as np
import talib as ta
from utils.logger import get_logger
from numba import njit

logger = get_logger(__name__)

# --- NEW: Numba JIT-compiled function for high-speed calculation ---
@njit
def _calculate_ce_signals_numba(close, long_stop, short_stop):
    """
    A Numba-accelerated function to compute Chandelier Exit signals.
    Operates on NumPy arrays for extreme speed.
    """
    n = len(close)
    ce_direction = np.zeros(n, dtype=np.int8)
    bullish_flip = np.zeros(n, dtype=np.int8)
    bearish_flip = np.zeros(n, dtype=np.int8)

    for i in range(1, n):
        # Determine trend direction
        if close[i] > long_stop[i-1]:
            ce_direction[i] = 1
        elif close[i] < short_stop[i-1]:
            ce_direction[i] = -1
        else:
            ce_direction[i] = ce_direction[i-1]

        # Detect flips
        if ce_direction[i] == 1 and ce_direction[i-1] == -1:
            bullish_flip[i] = 1

        if ce_direction[i] == -1 and ce_direction[i-1] == 1:
            bearish_flip[i] = 1
            
    return ce_direction, bullish_flip, bearish_flip


# utils/custom_indicators.py

# ... (نفس الكود للدالة _calculate_ce_signals_numba)

def _add_chandelier_exit_signals(df: pd.DataFrame, period=7, multiplier=4.0):
    """
    Calculates Chandelier Exit using a high-performance Numba backend
    AND adds a strategic "distance to stop" feature.
    """
    logger.info("Engineering Chandelier Exit signals and strategic distance feature...")

    atr = ta.ATR(df['high'], df['low'], df['close'], timeperiod=period)

    long_stop = df['high'].rolling(period).max() - atr * multiplier
    short_stop = df['low'].rolling(period).min() + atr * multiplier

    ce_direction, bullish_flip, bearish_flip = _calculate_ce_signals_numba(
        df['close'].values, 
        long_stop.values, 
        short_stop.values
    )
    
    df['ce_direction'] = ce_direction
    df['bullish_flip'] = bullish_flip
    df['bearish_flip'] = bearish_flip
    
    distance_to_stop = np.where(
        df['ce_direction'] == 1,
        (df['close'] - long_stop) / (df['close'] + 1e-9),
        (short_stop - df['close']) / (df['close'] + 1e-9)
    )
    df['dist_to_ce_stop_pct'] = distance_to_stop * 100
    # ------------------------------------
    
    return df

def _add_multi_timeframe_features(df: pd.DataFrame):
    """
    SCIENTIFICALLY-CORRECT MULTI-TIMEFRAME FEATURE ENGINEERING
    """
    if not isinstance(df.index, pd.DatetimeIndex): return df
    logger.info("Engineering multi-timeframe features (look-ahead bias corrected)...")
    timeframes = {'15min': '15min', '1h': '1h'}

    for name, rule in timeframes.items():
        df_resampled = df.resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()

        rsi_htf = ta.RSI(df_resampled['close'], timeperiod=14)
        adx_htf = ta.ADX(df_resampled['high'], df_resampled['low'], df_resampled['close'], timeperiod=14)

        df_indicators_shifted = pd.DataFrame({
            f'rsi_{name}': rsi_htf,
            f'adx_{name}': adx_htf
        }).shift(1)

        df = pd.merge(df, df_indicators_shifted, left_index=True, right_index=True, how='left')

        df[f'rsi_{name}'] = df[f'rsi_{name}'].ffill()
        df[f'adx_{name}'] = df[f'adx_{name}'].ffill()

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
    logger.info("Engineering advanced statistical features (Skew/Kurtosis)...")
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
    ad = ta.AD(hi, lo, cl, vol)
    df['cmf'] = ad.rolling(20).sum() / vol.rolling(20).sum()
    df['vi_plus'] = ta.PLUS_DI(hi, lo, cl)
    df['vi_minus'] = ta.MINUS_DI(hi, lo, cl)
    
    df = _add_multi_timeframe_features(df)
    df = _add_volatility_normalized_features(df)
    
    logger.info("Engineering feature interactions...")
    df['rsi_x_adx'] = (df['rsi'] / 100) * (df['adx'] / 100)
    
    df = _add_candlestick_features(df)
    df = _add_time_features(df)
    df = _add_advanced_statistical_features(df) 
    df = _add_chandelier_exit_signals(df) 

    logger.info("Cleaning up final dataframe...")
    return df.bfill().ffill().fillna(0)