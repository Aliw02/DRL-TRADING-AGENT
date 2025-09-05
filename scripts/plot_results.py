# scripts/plot_results.py
# HIERARCHICAL SQUAD PERFORMANCE AND INTELLIGENCE DASHBOARD

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import sys

# --- Add project root to path ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import paths
from utils.metrics import calculate_performance_metrics
from utils.logger import setup_logging, get_logger

class HierarchicalPlotter:
    def __init__(self, results_suffix=""):
        """
        Initializes the plotter for a specific backtest run (e.g., a stress test).
        """
        self.plot_dir = paths.RESULTS_DIR / f"plots{results_suffix}"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.results_suffix = results_suffix
        self.data_loaded = self._load_data()

    def _load_data(self):
        """ Robustly loads all data files required for plotting. """
        try:
            equity_path = paths.RESULTS_DIR / f"hierarchical_backtest_equity{self.results_suffix}.csv"
            trades_path = paths.RESULTS_DIR / f"hierarchical_backtest_trades{self.results_suffix}.csv"
            sim_path = paths.RESULTS_DIR / "final_backtest_simulation_data.csv" # Base sim data for price
            
            self.equity_df = pd.read_csv(equity_path)
            self.trades_df = pd.read_csv(trades_path, parse_dates=['entry_time', 'exit_time'])
            sim_df_raw = pd.read_csv(sim_path, parse_dates=['timestamp'])
            
            # Reconstruct equity curve with correct timestamps
            self.equity_df['timestamp'] = sim_df_raw['timestamp'].iloc[:len(self.equity_df)]
            self.sim_df = sim_df_raw
            return True
        except FileNotFoundError as e:
            print(f"ERROR: A required file for plotting was not found: {e}")
            return False

    def plot_equity_curve(self):
        """ Plots the main equity curve against a Buy & Hold benchmark. """
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(self.equity_df['timestamp'], self.equity_df['equity'], label='Hierarchical Squad Equity', color='navy', linewidth=2)
        initial_price = self.sim_df['close'].iloc[0]
        buy_hold_equity = self.sim_df['close'] * (self.equity_df['equity'].iloc[0] / initial_price)
        ax.plot(self.sim_df['timestamp'], buy_hold_equity, label='Buy & Hold Benchmark', linestyle='--', color='gray', linewidth=1.5)
        ax.set_title(f'Squad Performance: Equity Curve vs. Benchmark{self.results_suffix}', fontsize=18)
        ax.set_xlabel('Date'); ax.set_ylabel('Portfolio Value ($)'); ax.legend()
        plt.savefig(self.plot_dir / "equity_curve.png"); plt.close()

    def plot_hourly_performance(self):
        """ Plots profitable and losing trades by hour of the day. """
        if self.trades_df.empty: return
        self.trades_df['hour'] = self.trades_df['exit_time'].dt.hour
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
        sns.countplot(data=self.trades_df[self.trades_df['net_profit'] > 0], x='hour', ax=ax1, color='green')
        ax1.set_title('Profitable Trades by Hour'); ax1.set_xlabel('Hour of Day'); ax1.set_ylabel('Number of Trades')
        sns.countplot(data=self.trades_df[self.trades_df['net_profit'] <= 0], x='hour', ax=ax2, color='red')
        ax2.set_title('Losing Trades by Hour'); ax2.set_xlabel('Hour of Day'); ax2.set_ylabel('')
        plt.suptitle('Hourly Trading Performance', fontsize=16); plt.tight_layout()
        plt.savefig(self.plot_dir / "hourly_performance.png"); plt.close()

    def plot_daily_performance(self):
        """ Plots profitable and losing trades by day of the week. """
        if self.trades_df.empty: return
        self.trades_df['day'] = self.trades_df['exit_time'].dt.dayofweek
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
        sns.countplot(data=self.trades_df[self.trades_df['net_profit'] > 0], x='day', ax=ax1, color='green', order=range(7))
        ax1.set_title('Profitable Trades by Day'); ax1.set_xticklabels(day_names)
        sns.countplot(data=self.trades_df[self.trades_df['net_profit'] <= 0], x='day', ax=ax2, color='red', order=range(7))
        ax2.set_title('Losing Trades by Day'); ax2.set_xticklabels(day_names)
        plt.suptitle('Daily Trading Performance', fontsize=16); plt.tight_layout()
        plt.savefig(self.plot_dir / "daily_performance.png"); plt.close()
        
    def plot_specialist_squad_report(self):
        """
        THE STRATEGIC REPORT: Analyzes and visualizes the performance of each
        individual specialist agent in the squad.
        """
        if self.trades_df.empty or 'regime' not in self.trades_df.columns:
            print("Cannot generate squad report: 'regime' column not found in trades log.")
            return
            
        logger = get_logger(__name__)
        logger.info("Generating Specialist Squad Performance Report...")

        # Calculate metrics for each specialist
        squad_performance = self.trades_df.groupby('regime')['net_profit'].agg(
            total_pnl='sum',
            trade_count='count',
            avg_pnl='mean',
            win_rate=lambda x: (x > 0).mean()
        ).reset_index()

        # Identify key contributors
        biggest_winner_id = squad_performance.loc[squad_performance['total_pnl'].idxmax()]
        biggest_loser_id = squad_performance.loc[squad_performance['total_pnl'].idxmin()]
        most_active_id = squad_performance.loc[squad_performance['trade_count'].idxmax()]
        
        # --- Create the Dashboard ---
        fig = plt.figure(figsize=(20, 15), constrained_layout=True)
        fig.suptitle('Hierarchical Squad: Specialist Performance Report', fontsize=24)
        
        gs = fig.add_gridspec(2, 2)
        
        # 1. Total Profit/Loss by Specialist
        ax1 = fig.add_subplot(gs[0, 0])
        sns.barplot(data=squad_performance, x='regime', y='total_pnl', ax=ax1, palette='viridis')
        ax1.set_title('Total Net Profit by Specialist'); ax1.set_xlabel('Specialist (Regime ID)'); ax1.set_ylabel('Total PnL ($)')

        # 2. Activity Level by Specialist
        ax2 = fig.add_subplot(gs[0, 1])
        sns.barplot(data=squad_performance, x='regime', y='trade_count', ax=ax2, palette='coolwarm')
        ax2.set_title('Activity Level (Number of Trades)'); ax2.set_xlabel('Specialist (Regime ID)'); ax2.set_ylabel('Trade Count')

        # 3. Win Rate by Specialist
        ax3 = fig.add_subplot(gs[1, 0])
        sns.barplot(data=squad_performance, x='regime', y='win_rate', ax=ax3, palette='crest')
        ax3.set_title('Win Rate by Specialist'); ax3.set_xlabel('Specialist (Regime ID)'); ax3.set_ylabel('Win Rate (%)')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        # 4. Average PnL per Trade
        ax4 = fig.add_subplot(gs[1, 1])
        sns.barplot(data=squad_performance, x='regime', y='avg_pnl', ax=ax4, palette='magma')
        ax4.set_title('Average Profit/Loss per Trade'); ax4.set_xlabel('Specialist (Regime ID)'); ax4.set_ylabel('Avg PnL ($)')

        plt.savefig(self.plot_dir / "specialist_squad_report.png"); plt.close()
        logger.info(f"Squad report saved to {self.plot_dir / 'specialist_squad_report.png'}")


    def run_all_plots(self):
        if not self.data_loaded: return
        print("\nGenerating all strategic plots...")
        self.plot_equity_curve()
        self.plot_hourly_performance()
        self.plot_daily_performance()
        self.plot_specialist_squad_report() # Add the new report to the main call
        print("\n✅ All plots generated successfully.")

if __name__ == "__main__":
    # To run for a specific stress test, you would pass the suffix
    # e.g., plotter = HierarchicalPlotter(results_suffix="_stresstest_flash_crash")
    plotter = HierarchicalPlotter()
    plotter.run_all_plots()