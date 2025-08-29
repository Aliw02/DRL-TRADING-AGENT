# Performance metrics calculation
import numpy as np
import pandas as pd

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    """
    Calculate the Sharpe Ratio
    """
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)

def calculate_max_drawdown(equity_curve):
    """
    Calculate the Maximum Drawdown
    """
    peak = equity_curve[0]
    max_drawdown = 0
    for x in equity_curve:
        if x > peak:
            peak = x
        drawdown = (peak - x) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown

def calculate_sortino_ratio(returns, risk_free_rate=0.01):
    """
    Calculate the Sortino Ratio
    """
    downside_returns = returns[returns < risk_free_rate]
    if downside_returns.empty:
        return np.nan
    return (np.mean(returns) - risk_free_rate) / np.std(downside_returns)

def calculate_performance_metrics(equity_curve, risk_free_rate=0.01):
    """
    Calculate all performance metrics
    """
    returns = equity_curve.pct_change().dropna()
    metrics = {
        "sharpe_ratio": calculate_sharpe_ratio(returns, risk_free_rate),
        "max_drawdown": calculate_max_drawdown(equity_curve),
        "sortino_ratio": calculate_sortino_ratio(returns, risk_free_rate)
    }
    return pd.Series(metrics)