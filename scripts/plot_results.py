# scripts/plot_results.py (FINAL, ROBUST, AND COMPLETE VERSION)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import os
import sys

# --- Add project path ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import paths

sns.set(style='whitegrid', font_scale=1.2)

class Plotter:
    def __init__(self):
        """
        Initializes the Plotter. It robustly loads all data files and reconstructs
        the equity DataFrame to ensure it has the correct timestamps.
        """
        self.plot_dir = paths.RESULTS_DIR / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        self.equity_df = None
        self.trades_df = None
        self.sim_df = None
        self.data_loaded = False

        try:
            # --- Step 1: Attempt to load all necessary files ---
            equity_path = paths.RESULTS_DIR / "final_backtest_equity.csv"
            trades_path = paths.RESULTS_DIR / "final_backtest_trades.csv"
            sim_path = paths.RESULTS_DIR / "final_backtest_simulation_data.csv"

            print("Attempting to load backtest result files...")
            equity_data = pd.read_csv(equity_path)
            self.trades_df = pd.read_csv(trades_path)
            self.sim_df = pd.read_csv(sim_path)
            
            # --- Step 2: If loading is successful, robustly process the data ---
            print("Files loaded. Processing and aligning data...")
            
            # Convert date columns for trades and simulation data
            self.trades_df['entry_time'] = pd.to_datetime(self.trades_df['entry_time'])
            self.trades_df['exit_time'] = pd.to_datetime(self.trades_df['exit_time'])
            self.sim_df['timestamp'] = pd.to_datetime(self.sim_df['timestamp'])

            # --- THE DEFINITIVE FIX FOR THE EQUITY CURVE ---
            # Reconstruct the equity DataFrame correctly.
            # The timestamps should come from the simulation data, as the equity curve
            # corresponds to the bars that were actually simulated.
            equity_values = equity_data.iloc[:, 0].values
            # The number of equity points matches the number of simulated bars
            num_simulated_bars = len(self.sim_df)
            # The equity curve starts from the first bar of the simulation
            self.equity_df = pd.DataFrame({
                'timestamp': self.sim_df['timestamp'].iloc[:len(equity_values)],
                'equity': equity_values
            })

            print("All necessary data for plotting is ready.")
            self.data_loaded = True

        except FileNotFoundError as e:
            print(f"ERROR: A required file for plotting was not found: {e}")
            print("Please ensure you have run the backtest_agent.py script successfully.")
        except Exception as e:
            print(f"An unexpected error occurred during data loading: {e}")

    # --- Helper functions (no changes) ---
    def _plot_hourly_histogram(self, df: pd.DataFrame, title: str, filename: str):
        if df.empty: return
        df['hour'] = df['exit_time'].dt.hour
        plt.figure(figsize=(12, 7)); df.groupby('hour').size().plot(kind='bar', color='#4c72b0')
        plt.title(title, fontsize=16); plt.xlabel('Hour of the Day'); plt.ylabel('Number of Trades')
        plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(self.plot_dir / filename); plt.close()

    def _plot_daily_histogram(self, df: pd.DataFrame, title: str, filename: str):
        if df.empty: return
        df['day'] = df['exit_time'].dt.dayofweek
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        plt.figure(figsize=(12, 7)); df.groupby('day').size().reindex(range(7), fill_value=0).plot(kind='bar', color='#55a868')
        plt.title(title, fontsize=16); plt.xlabel('Day of the Week'); plt.ylabel('Number of Trades')
        plt.xticks(range(7), day_names, rotation=0); plt.tight_layout(); plt.savefig(self.plot_dir / filename); plt.close()

    # --- Main plotting functions ---
    def plot_equity_curve_simple(self):
        plt.figure(figsize=(15, 8))
        # Use the corrected self.equity_df
        plt.plot(self.equity_df['timestamp'], self.equity_df['equity'], label='Equity', color='teal', linewidth=2.5)
        
        initial_price = self.sim_df['close'].iloc[0]
        buy_hold_equity = self.sim_df['close'] * (self.equity_df['equity'].iloc[0] / initial_price)
        plt.plot(self.sim_df['timestamp'], buy_hold_equity, label='Buy & Hold Benchmark', linestyle='--', color='gray', linewidth=2)
        
        plt.title('Agent Performance: Equity Curve vs. Buy & Hold', fontsize=16); plt.xlabel('Date'); plt.ylabel('Portfolio Value ($)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.legend(); plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate(); plt.tight_layout(); plt.savefig(self.plot_dir / "equity_curve.png"); plt.close()

    def plot_performance_metrics(self):
        from utils.metrics import calculate_performance_metrics
        performance_stats = calculate_performance_metrics(self.equity_df['equity'])
        metrics = {'Sharpe Ratio': performance_stats['sharpe_ratio'], 'Sortino Ratio': performance_stats['sortino_ratio'],
                   'Calmar Ratio': performance_stats['calmar_ratio'], 'Max Drawdown': performance_stats['max_drawdown']}
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics.keys(), metrics.values(), color=['#2a7e78', '#64a6a1', '#9bc2c0', '#d95f54'])
        plt.title('Key Performance Metrics', fontsize=16); plt.ylabel('Value')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', ha='center', va='bottom')
        plt.tight_layout(); plt.savefig(self.plot_dir / "performance_metrics.png"); plt.close()

    def plot_profitable_trades_hourly(self):
        self._plot_hourly_histogram(self.trades_df[self.trades_df['net_profit'] > 0].copy(), 'Number of Profitable Trades by Hour of Day', 'profitable_trades_hourly.png')

    def plot_losing_trades_hourly(self):
        self._plot_hourly_histogram(self.trades_df[self.trades_df['net_profit'] < 0].copy(), 'Number of Losing Trades by Hour of Day', 'losing_trades_hourly.png')
        
    def plot_profitable_trades_daily(self):
        self._plot_daily_histogram(self.trades_df[self.trades_df['net_profit'] > 0].copy(), 'Number of Profitable Trades by Day of Week', 'profitable_trades_daily.png')
        
    def plot_losing_trades_daily(self):
        self._plot_daily_histogram(self.trades_df[self.trades_df['net_profit'] < 0].copy(), 'Number of Losing Trades by Day of Week', 'losing_trades_daily.png')

    def plot_detailed_equity_vs_price(self):
        print("Generating detailed price vs. equity performance chart...")
        fig, ax1 = plt.subplots(figsize=(20, 10)); fig.suptitle('Detailed Performance: Price Action vs. Equity Curve', fontsize=20)
        ax1.set_xlabel('Date', fontsize=16); ax1.set_ylabel('Price (XAUUSD)', color='teal', fontsize=16)
        ax1.plot(self.sim_df['timestamp'], self.sim_df['close'], color='teal', label='Close Price', alpha=0.8, linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor='teal')
        buy_trades = self.trades_df[self.trades_df['type'] == 'BUY']; sell_trades = self.trades_df[self.trades_df['type'] == 'SELL']
        ax1.plot(buy_trades['entry_time'], buy_trades['entry_price'], '^', color='green', markersize=8, label='Buy Entry')
        ax1.plot(sell_trades['entry_time'], sell_trades['entry_price'], 'v', color='red', markersize=8, label='Sell Entry')
        ax1.legend(loc='upper left'); ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2 = ax1.twinx(); ax2.set_ylabel('Portfolio Value ($)', color='navy', fontsize=16)
        # Use the corrected self.equity_df
        ax2.plot(self.equity_df['timestamp'], self.equity_df['equity'], color='navy', label='Equity Curve', linewidth=2.0)
        ax2.tick_params(axis='y', labelcolor='navy'); ax2.legend(loc='upper right')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M')); fig.autofmt_xdate()
        plt.tight_layout(rect=[0, 0, 1, 0.96]); save_path = self.plot_dir / "detailed_performance_chart.png"
        plt.savefig(save_path); plt.close(); print(f"Detailed chart saved to: {save_path}")

    def run_all_plots(self):
        """A single function to run all available plots only if data is loaded."""
        if not self.data_loaded:
            print("Cannot generate plots because data was not loaded successfully.")
            return

        print("\nGenerating all plots...")
        try:
            self.plot_equity_curve_simple()
            self.plot_performance_metrics()
            self.plot_profitable_trades_hourly()
            self.plot_losing_trades_hourly()
            self.plot_profitable_trades_daily()
            self.plot_losing_trades_daily()
            self.plot_detailed_equity_vs_price()
            print("\n✅ All plots generated successfully.")
        except Exception as e:
            print(f"An error occurred during plotting: {e}")


if __name__ == "__main__":
    plotter = Plotter()
    plotter.run_all_plots()