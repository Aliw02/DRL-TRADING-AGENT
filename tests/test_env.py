# Tests for trading environment
import pytest
import pandas as pd
import numpy as np
from envs.trading_env import TradingEnv

def create_test_data():
    """Create test data for the environment"""
    np.random.seed(42)
    n_samples = 200
    
    # Create basic price data
    prices = np.cumsum(np.random.randn(n_samples) * 0.1) + 100
    
    # Create test DataFrame with all required features
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.rand(n_samples) * 0.5,
        'low': prices - np.random.rand(n_samples) * 0.5,
        'close': prices,
        'volume': np.random.randint(100, 1000, n_samples)
    })
    
    # Add all the required indicator columns from config
    feature_columns = [
        'direction', 'bullish_flip', 'bearish_flip', 'dist_to_stop',
        'adx', 'rsi', 'plus_di', 'minus_di', 'ma_fast', 'ma_slow',
        'macd', 'macd_signal', 'bb_width', 'roc', 'volatility', 'volume_ratio',
        'dist_to_fib_0.382', 'dist_to_fib_0.618', 'rsi_x_adx', 'market_regime',
        'body_to_wick_ratio', 'hour_of_day', 'day_of_week', 'rsi_change', 'dist_from_ma_slow'
    ]
    for col in feature_columns:
        if col in ['hour_of_day', 'day_of_week']:
            df[col] = np.random.randint(0, 24 if col == 'hour_of_day' else 7, n_samples)
        else:
            df[col] = np.random.randn(n_samples)
    
    return df

def test_env_creation():
    """Test that the environment can be created"""
    df = create_test_data()
    env = TradingEnv(df, initial_balance=100, sequence_length=150)
    
    # Check that environment has correct attributes
    assert env.initial_balance == 100
    assert env.sequence_length == 150

def test_env_reset():
    """Test environment reset"""
    df = create_test_data()
    env = TradingEnv(df)
    
    obs, info = env.reset()
    
    # Check that observation has correct shape (25 features + position + equity)
    assert obs.shape == (150, 27)  # 25 features + position + equity
    
    # Check initial state
    assert env.position == 0
    assert env.equity == 100
    assert env.current_step == 150

def test_env_step():
    """Test environment step function"""
    df = create_test_data()
    env = TradingEnv(df)
    obs, info = env.reset()
    
    # Take a step
    next_obs, reward, done, truncated, info = env.step(0)  # Hold action
    
    # Check that observation shape is maintained
    assert next_obs.shape == (150, 27)
    
    # Check that step counter increased
    assert env.current_step == 151
    
    # Check info dict
    assert 'equity' in info

def test_env_actions():
    """Test different environment actions"""
    df = create_test_data()
    env = TradingEnv(df)
    obs, info = env.reset()
    
    # Test buy action
    next_obs, reward, done, truncated, info = env.step(1)  # Buy
    assert env.position == 1
    
    # Test sell action
    next_obs, reward, done, truncated, info = env.step(2)  # Sell
    assert env.position == 0
    
    # Test hold action
    next_obs, reward, done, truncated, info = env.step(0)  # Hold
    assert env.position == 0

def test_env_done():
    """Test environment done condition"""
    df = create_test_data()
    env = TradingEnv(df)
    obs, info = env.reset()
    
    # Step until done
    done = False
    steps = 0
    while not done and steps < 100:
        next_obs, reward, done, truncated, info = env.step(0)  # Hold
        steps += 1
    
    # Should eventually be done
    assert done

if __name__ == "__main__":
    pytest.main([__file__, "-v"])