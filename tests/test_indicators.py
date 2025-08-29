# Tests for indicators
import pytest
import pandas as pd
import numpy as np
from utils.custom_indicators import calculate_rsi, calculate_adx, calculate_chandelier_exit, calculate_all_indicators

def test_rsi_calculation():
    """Test RSI calculation with known values"""
    # Create test data
    prices = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84])
    rsi = calculate_rsi(prices, period=3)
    
    # RSI should be between 0 and 100
    assert np.all(rsi >= 0) and np.all(rsi <= 100)
    assert not np.any(np.isnan(rsi))

def test_adx_calculation():
    """Test ADX calculation with known values"""
    # Create test data
    high = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    low = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    close = np.array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=3)
    
    # ADX and DI should be reasonable values
    assert np.all(adx >= 0) and np.all(adx <= 100)
    assert np.all(plus_di >= 0) and np.all(plus_di <= 100)
    assert np.all(minus_di >= 0) and np.all(minus_di <= 100)

def test_chandelier_exit():
    """Test Chandelier Exit calculation"""
    # Create test data
    high = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    low = pd.Series([8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    close = pd.Series([9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    
    direction, bullish_flip, bearish_flip, dist_to_stop = calculate_chandelier_exit(high, low, close)
    
    # Direction should be 1 or -1
    assert np.all(np.isin(direction, [1, -1]))
    # Flips should be 0 or 1
    assert np.all(np.isin(bullish_flip, [0, 1]))
    assert np.all(np.isin(bearish_flip, [0, 1]))

def test_all_indicators():
    """Test calculation of all indicators"""
    # Create test DataFrame
    df = pd.DataFrame({
        'open': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        'high': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'low': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        'close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        'volume': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    })
    
    # Calculate all indicators
    result = calculate_all_indicators(df)
    
    # Should return a DataFrame with the expected columns
    expected_columns = [
        'direction', 'bullish_flip', 'bearish_flip', 'dist_to_stop',
        'adx', 'rsi', 'plus_di', 'minus_di', 'ma_fast', 'ma_slow',
        'macd', 'macd_signal', 'bb_width', 'roc', 'volatility', 'volume_ratio'
    ]
    
    assert all(col in result.columns for col in expected_columns)
    assert len(result) == len(df)
    assert not result.isnull().values.any()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])