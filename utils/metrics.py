# utils/metrics.py (UPGRADED)
import numpy as np
import pandas as pd

def calculate_sharpe_ratio(returns, periods_per_year=252*24*60):
    if np.std(returns) == 0: return 0
    annualized_mean_return = np.mean(returns) * periods_per_year
    annualized_return_std = np.std(returns) * np.sqrt(periods_per_year)
    return annualized_mean_return / annualized_return_std

def calculate_sortino_ratio(returns, periods_per_year=252*24*60):
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or np.std(downside_returns) == 0: return 0
    expected_return = np.mean(returns) * periods_per_year
    downside_std = np.std(downside_returns) * np.sqrt(periods_per_year)
    return expected_return / downside_std

def calculate_calmar_ratio(equity_curve, periods_per_year=252*24*60):
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0: return 0
    max_drawdown = calculate_max_drawdown(equity_curve)
    if max_drawdown == 0: return np.inf
    annualized_return = np.mean(returns) * periods_per_year
    return annualized_return / max_drawdown

def calculate_max_drawdown(equity_curve):
    peak = equity_curve.iloc[0]
    max_drawdown = 0
    for equity in equity_curve:
        if equity > peak: peak = equity
        drawdown = (peak - equity) / peak
        if drawdown > max_drawdown: max_drawdown = drawdown
    return max_drawdown

def calculate_performance_metrics(equity_curve, periods_per_year=252*24*15):
    if isinstance(equity_curve, list): equity_curve = pd.Series(equity_curve)
    returns = equity_curve.pct_change().dropna()
    if returns.empty:
        return pd.Series({"sharpe_ratio": 0, "sortino_ratio": 0, "calmar_ratio": 0,
                          "max_drawdown": calculate_max_drawdown(equity_curve)})
    metrics = {"sharpe_ratio": calculate_sharpe_ratio(returns, periods_per_year),
               "sortino_ratio": calculate_sortino_ratio(returns, periods_per_year),
               "calmar_ratio": calculate_calmar_ratio(equity_curve, periods_per_year),
               "max_drawdown": calculate_max_drawdown(equity_curve)}
    return pd.Series(metrics)