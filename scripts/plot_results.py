"""Results plotting utilities for walk-forward training.

This module exposes a function `analyze_and_plot_all_results(all_results, save_dir)`
which accepts the `all_results` list of dicts produced by `run_walk_forward_training()`
and generates summary plots and a CSV summary.

The function is robust: it will detect numeric columns in the results and plot
them across splits. It also saves plots to `save_dir` and returns the summary DataFrame.
"""

import os
from typing import List, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')


# plotter.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Plotter:
    def __init__(self, equity_path="results/backtest_equity.csv", trades_path="results/backtest_trades.csv"):
        try:
            self.equity_df = pd.read_csv(equity_path)
            self.trades_df = pd.read_csv(trades_path)
            self.trades_df['entry_time'] = pd.to_datetime(self.trades_df['entry_time'])
            self.trades_df['exit_time'] = pd.to_datetime(self.trades_df['exit_time'])
            print("Data loaded successfully.")
        except FileNotFoundError:
            print("please run backtester.py first.")
            self.equity_df = None
            self.trades_df = None

    def plot_equity_curve(self):
        """1. يرسم منحنى الرصيد على مدى الوقت."""
        if self.equity_df is None: return
        
        self.equity_df['timestamp'] = pd.to_datetime(self.equity_df['timestamp'])
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_df['timestamp'], self.equity_df['equity'], label='Equity')
        plt.title('Equity Curve During Backtesting')
        plt.xlabel('Time')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
        plt.gcf().autofmt_xdate()
        plt.show()

    def _plot_hourly_histogram(self, df: pd.DataFrame, title: str):
        """دالة مساعدة لرسم مخطط الساعات."""
        if df.empty:
            print("There is no data to plot.")
            return

        df['hour'] = df['exit_time'].dt.hour
        df['dayofweek'] = df['exit_time'].dt.dayofweek
        
        # إعادة تسمية أيام الأسبوع باللغة العربية
        day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        df['day_name'] = df['dayofweek'].map(day_names)
        
        df_grouped = df.groupby('hour').size()
        plt.figure(figsize=(10, 6))
        df_grouped.plot(kind='bar')
        plt.title(title)
        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Trades') 
        plt.xticks(rotation=0)
        plt.grid(axis='y')
        plt.show()

    def plot_profitable_hours(self):
        """2. يرسم مخطط الساعات الأكثر ربحية."""
        profitable_trades = self.trades_df[self.trades_df['profit_loss'] > 0]
        self._plot_hourly_histogram(profitable_trades, 'Counter of the Profitable Trades by Hour')

    def plot_losing_hours(self):
        """3. يرسم مخطط الساعات الأكثر خسارة."""
        losing_trades = self.trades_df[self.trades_df['profit_loss'] < 0]
        self._plot_hourly_histogram(losing_trades, 'Counter of the Losing Trades by Hour')

    def _plot_daily_histogram(self, df: pd.DataFrame, title: str):
        """دالة مساعدة لرسم مخطط الأيام."""
        if df.empty:
            print("There is no data to plot.")
            return

        df['day'] = df['exit_time'].dt.dayofweek
        df_grouped = df.groupby('day').size().reindex(range(7), fill_value=0)

        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        plt.figure(figsize=(10, 6))
        df_grouped.plot(kind='bar')
        plt.title(title)
        plt.xlabel('Day of the Week')
        plt.ylabel('Number of Trades')
        plt.xticks(range(7), day_names, rotation=0)
        plt.grid(axis='y')
        plt.show()
    
    def plot_profitable_days(self):
        """4. يرسم مخطط الأيام الأكثر ربحية."""
        profitable_trades = self.trades_df[self.trades_df['profit_loss'] > 0]
        self._plot_daily_histogram(profitable_trades, 'Number of Profitable Trades by Day')

    def plot_losing_days(self):
        """5. يرسم مخطط الأيام الأكثر خسارة."""
        losing_trades = self.trades_df[self.trades_df['profit_loss'] < 0]
        self._plot_daily_histogram(losing_trades, 'Number of Losing Trades by Day')
        
if __name__ == "__main__":
    plotter = Plotter()
    if plotter.equity_df is not None and plotter.trades_df is not None:
        plotter.plot_equity_curve()
        plotter.plot_profitable_hours()
        plotter.plot_losing_hours()
        plotter.plot_profitable_days()
        plotter.plot_losing_days()


