# scripts/plot_results.py (FINAL UPGRADED VERSION)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import os
import sys

# --- Add project path to allow imports from other directories ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.metrics import calculate_performance_metrics
from config import paths

sns.set(style='whitegrid', font_scale=1.1)

class Plotter:
    def __init__(self, equity_path=paths.RESULTS_DIR / "final_backtest_equity.csv", trades_path=paths.RESULTS_DIR / "final_backtest_trades.csv"):
        # Create the plots directory if it doesn't exist
        self.plot_dir = paths.RESULTS_DIR / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.equity_df = pd.read_csv(equity_path)
            self.trades_df = pd.read_csv(trades_path)
            
            # Ensure 'timestamp' column exists and is in datetime format
            if 'timestamp' not in self.equity_df.columns:
                self.equity_df.reset_index(inplace=True)
                self.equity_df.rename(columns={'index': 'timestamp'}, inplace=True)
            self.equity_df['timestamp'] = pd.to_datetime(self.equity_df['timestamp'])

            if 'entry_time' in self.trades_df.columns:
                self.trades_df['entry_time'] = pd.to_datetime(self.trades_df['entry_time'])
                self.trades_df['exit_time'] = pd.to_datetime(self.trades_df['exit_time'])
            
            print("Data loaded successfully.")
        except FileNotFoundError:
            print("Please run backtest_agent.py first to generate the necessary files.")
            self.equity_df = None
            self.trades_df = None
            
    def _plot_hourly_histogram(self, df: pd.DataFrame, title: str, filename: str):
        """Helper function to plot hourly histogram."""
        if df.empty:
            print(f"There is no data to plot for: {title}")
            return
        
        df['hour'] = df['exit_time'].dt.hour
        df_grouped = df.groupby('hour').size()
        plt.figure(figsize=(10, 6))
        df_grouped.plot(kind='bar', color='#4c72b0')
        plt.title(title, fontsize=16)
        plt.xlabel('Hour of the Day', fontsize=14)
        plt.ylabel('Number of Trades', fontsize=14) 
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(self.plot_dir / filename)
        plt.close()

    def _plot_daily_histogram(self, df: pd.DataFrame, title: str, filename: str):
        """Helper function to plot daily histogram."""
        if df.empty:
            print(f"There is no data to plot for: {title}")
            return
        
        df['day'] = df['exit_time'].dt.dayofweek
        df_grouped = df.groupby('day').size().reindex(range(7), fill_value=0)

        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        plt.figure(figsize=(10, 6))
        df_grouped.plot(kind='bar', color='#55a868')
        plt.title(title, fontsize=16)
        plt.xlabel('Day of the Week', fontsize=14)
        plt.ylabel('Number of Trades', fontsize=14)
        plt.xticks(range(7), day_names, rotation=0)
        plt.grid(axis='y', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(self.plot_dir / filename)
        plt.close()

    def plot_equity_curve(self):
        """1. Plots the equity curve over time."""
        if self.equity_df is None: return
        
        plt.figure(figsize=(15, 8))
        plt.plot(self.equity_df['timestamp'], self.equity_df['0'], label='Equity', color='teal', linewidth=2.5)
        
        initial_price = self.equity_df.iloc[0]['0']
        final_price = self.equity_df.iloc[-1]['0']
        plt.plot(self.equity_df['timestamp'], np.linspace(initial_price, final_price, len(self.equity_df)), 
                 label='Buy & Hold Benchmark', linestyle='--', color='gray', linewidth=2)
        
        plt.title('Agent Performance: Equity Curve vs. Buy & Hold', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Portfolio Value ($)', fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(fontsize=12)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.savefig(self.plot_dir / "equity_curve.png")
        plt.close()

    def plot_performance_metrics(self):
        """2. Creates a bar chart for key performance metrics."""
        if self.equity_df is None: return

        returns = self.equity_df['0'].pct_change().dropna()
        if returns.empty:
            print("Not enough data to calculate performance metrics.")
            return

        performance_stats = calculate_performance_metrics(self.equity_df['0'])

        metrics = {
            'Sharpe Ratio': performance_stats['sharpe_ratio'],
            'Sortino Ratio': performance_stats['sortino_ratio'],
            'Calmar Ratio': performance_stats['calmar_ratio'],
            'Max Drawdown': performance_stats['max_drawdown']
        }
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_names, metric_values, color=['#2a7e78', '#64a6a1', '#9bc2c0', '#2a7e78'])
        plt.title('Key Performance Metrics', fontsize=16)
        plt.ylabel('Value', fontsize=14)
        plt.grid(axis='y', linestyle='--', linewidth=0.5)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=12)

        plt.tight_layout()
        plt.savefig(self.plot_dir / "performance_metrics.png")
        plt.close()

    def plot_profitable_trades_hourly(self):
        """3. Plots hourly distribution of profitable trades."""
        if self.trades_df is None or self.trades_df.empty: return
        profitable_trades = self.trades_df[self.trades_df['net_profit'] > 0].copy()
        self._plot_hourly_histogram(profitable_trades, 'Number of Profitable Trades by Hour of Day', 'profitable_trades_hourly.png')

    def plot_losing_trades_hourly(self):
        """4. Plots hourly distribution of losing trades."""
        if self.trades_df is None or self.trades_df.empty: return
        losing_trades = self.trades_df[self.trades_df['net_profit'] < 0].copy()
        self._plot_hourly_histogram(losing_trades, 'Number of Losing Trades by Hour of Day', 'losing_trades_hourly.png')
        
    def plot_profitable_days(self):
        """5. Plots daily distribution of profitable trades."""
        if self.trades_df is None or self.trades_df.empty: return
        profitable_trades = self.trades_df[self.trades_df['net_profit'] > 0].copy()
        self._plot_daily_histogram(profitable_trades, 'Number of Profitable Trades by Day of Week', 'profitable_trades_daily.png')
        
    def plot_losing_days(self):
        """6. Plots daily distribution of losing trades."""
        if self.trades_df is None or self.trades_df.empty: return
        losing_trades = self.trades_df[self.trades_df['net_profit'] < 0].copy()
        self._plot_daily_histogram(losing_trades, 'Number of Losing Trades by Day of Week', 'losing_trades_daily.png')

if __name__ == "__main__":
    plotter = Plotter()
    if plotter.equity_df is not None:
        plotter.plot_equity_curve()
        plotter.plot_performance_metrics()
    if plotter.trades_df is not None and not plotter.trades_df.empty:
        plotter.plot_profitable_trades_hourly()
        plotter.plot_losing_trades_hourly()
        plotter.plot_profitable_days()
        plotter.plot_losing_days()