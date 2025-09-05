# scripts/plot_results.py
# HIERARCHICAL SQUAD PERFORMANCE AND INTELLIGENCE DASHBOARD (FINAL, HARDENED VERSION)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import sys

# --- Add project root to path for robust imports ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import paths
from utils.metrics import calculate_performance_metrics
from utils.logger import get_logger

class HierarchicalPlotter:
    """
    A comprehensive visualization suite for analyzing the performance of the
    hierarchical squad of specialist agents. It is hardened to handle cases
    with no trading activity to prevent pipeline failures.
    """
    def __init__(self, results_suffix=""):
        """
        Initializes the plotter for a specific backtest run (e.g., a stress test).
        """
        self.plot_dir = paths.RESULTS_DIR / f"plots{results_suffix}"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.results_suffix = results_suffix
        self.data_loaded = self._load_data()

    def _load_data(self) -> bool:
        """
        Robustly loads all data files required for plotting. Handles cases where
        the trades file is empty by creating a placeholder DataFrame.
        """
        logger = get_logger(__name__)
        try:
            equity_path = paths.RESULTS_DIR / f"hierarchical_backtest_equity{self.results_suffix}.csv"
            trades_path = paths.RESULTS_DIR / f"hierarchical_backtest_trades{self.results_suffix}.csv"
            # We use a base simulation data file for the price series benchmark
            sim_path = paths.RESULTS_DIR / "final_backtest_simulation_data.csv"
            
            self.equity_df = pd.read_csv(equity_path)
            
            # --- CRITICAL HARDENING FIX ---
            if os.path.exists(trades_path) and os.path.getsize(trades_path) > 5: # Check if file has more than just a header
                self.trades_df = pd.read_csv(trades_path, parse_dates=['entry_time', 'exit_time'])
            else:
                # If file is empty or doesn't exist, create a placeholder empty DataFrame
                logger.warning("Trades file is empty or not found. No trade-specific plots will be generated.")
                self.trades_df = pd.DataFrame(columns=['entry_time', 'exit_time', 'net_profit', 'regime'])

            sim_df_raw = pd.read_csv(sim_path, parse_dates=['timestamp'])
            
            # Reconstruct equity curve with correct timestamps from simulation data
            self.equity_df['timestamp'] = sim_df_raw['timestamp'].iloc[:len(self.equity_df)]
            self.sim_df = sim_df_raw

            logger.info("All necessary data for plotting is loaded and validated.")
            return True
        except FileNotFoundError as e:
            logger.error(f"A required file for plotting was not found: {e}. Plotting will be skipped.")
            return False
        except Exception as e:
            logger.error(f"A critical error occurred during data loading for plots: {e}", exc_info=True)
            return False

    def plot_equity_curve(self):
        """ Plots the main equity curve against a Buy & Hold benchmark and visualizes drawdown periods. """
        logger = get_logger(__name__)
        logger.info("Generating main equity curve plot...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8))
        
        ax.plot(self.equity_df['timestamp'], self.equity_df['equity'], label='Hierarchical Squad Equity', color='navy', linewidth=2.5)

        initial_price = self.sim_df['close'].iloc[0]
        buy_hold_equity = self.sim_df['close'] * (self.equity_df['equity'].iloc[0] / initial_price)
        ax.plot(self.sim_df['timestamp'], buy_hold_equity, label='Buy & Hold Benchmark', linestyle='--', color='gray', linewidth=1.5)
        
        high_water_mark = self.equity_df['equity'].cummax()
        ax.fill_between(self.equity_df['timestamp'], self.equity_df['equity'], high_water_mark, 
                        where=(self.equity_df['equity'] < high_water_mark),
                        color='red', alpha=0.2, interpolate=True, label='Drawdown Period')

        ax.set_title(f'Squad Performance: Equity Curve vs. Benchmark{self.results_suffix}', fontsize=18, weight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(self.plot_dir / "equity_curve.png")
        plt.close()

    def plot_hourly_performance(self):
        """ Plots profitable and losing trades by hour of the day. """
        if self.trades_df.empty: return
        logger = get_logger(__name__)
        logger.info("Generating hourly performance analysis plot...")
        self.trades_df['hour'] = self.trades_df['exit_time'].dt.hour
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
        
        sns.countplot(data=self.trades_df[self.trades_df['net_profit'] > 0], x='hour', ax=ax1, color='green', order=range(24))
        ax1.set_title('Profitable Trades by Hour', fontsize=14, weight='bold')
        ax1.set_xlabel('Hour of Day'); ax1.set_ylabel('Number of Trades')
        
        sns.countplot(data=self.trades_df[self.trades_df['net_profit'] <= 0], x='hour', ax=ax2, color='red', order=range(24))
        ax2.set_title('Losing Trades by Hour', fontsize=14, weight='bold')
        ax2.set_xlabel('Hour of Day'); ax2.set_ylabel('')
        
        plt.suptitle('Hourly Trading Performance Analysis', fontsize=18, weight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.plot_dir / "hourly_performance.png")
        plt.close()

    def plot_daily_performance(self):
        """ Plots profitable and losing trades by day of the week. """
        if self.trades_df.empty: return
        logger = get_logger(__name__)
        logger.info("Generating daily performance analysis plot...")
        self.trades_df['day'] = self.trades_df['exit_time'].dt.dayofweek
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
        
        sns.countplot(data=self.trades_df[self.trades_df['net_profit'] > 0], x='day', ax=ax1, color='green', order=range(7))
        ax1.set_title('Profitable Trades by Day', fontsize=14, weight='bold')
        ax1.set_xticklabels(day_names, rotation=45); ax1.set_xlabel('')
        
        sns.countplot(data=self.trades_df[self.trades_df['net_profit'] <= 0], x='day', ax=ax2, color='red', order=range(7))
        ax2.set_title('Losing Trades by Day', fontsize=14, weight='bold')
        ax2.set_xticklabels(day_names, rotation=45); ax2.set_xlabel('')
        
        plt.suptitle('Daily Trading Performance Analysis', fontsize=18, weight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.plot_dir / "daily_performance.png")
        plt.close()
        
    def plot_specialist_squad_report(self):
        """
        THE STRATEGIC REPORT: Analyzes and visualizes the performance of each
        individual specialist agent in the squad, providing deep intelligence.
        """
        if self.trades_df.empty or 'regime' not in self.trades_df.columns:
            return
            
        logger = get_logger(__name__)
        logger.info("Generating Specialist Squad Intelligence Report...")

        squad_performance = self.trades_df.groupby('regime')['net_profit'].agg(
            total_pnl='sum',
            trade_count='count',
            avg_pnl='mean',
            win_rate=lambda x: (x > 0).mean() if not x.empty else 0
        ).reset_index()

        fig = plt.figure(figsize=(20, 16), constrained_layout=True)
        fig.suptitle('Hierarchical Squad: Specialist Intelligence Report', fontsize=24, weight='bold')
        gs = fig.add_gridspec(2, 2)
        
        ax1 = fig.add_subplot(gs[0, 0])
        sns.barplot(data=squad_performance, x='regime', y='total_pnl', ax=ax1, palette='viridis')
        ax1.set_title('Total Net Profit by Specialist'); ax1.set_xlabel('Specialist (Regime ID)'); ax1.set_ylabel('Total PnL ($)')

        ax2 = fig.add_subplot(gs[0, 1])
        sns.barplot(data=squad_performance, x='regime', y='trade_count', ax=ax2, palette='coolwarm')
        ax2.set_title('Activity Level (Number of Trades)'); ax2.set_xlabel('Specialist (Regime ID)'); ax2.set_ylabel('Trade Count')

        ax3 = fig.add_subplot(gs[1, 0])
        sns.barplot(data=squad_performance, x='regime', y='win_rate', ax=ax3, palette='crest')
        ax3.set_title('Win Rate by Specialist'); ax3.set_xlabel('Specialist (Regime ID)'); ax3.set_ylabel('Win Rate')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        ax4 = fig.add_subplot(gs[1, 1])
        sns.barplot(data=squad_performance, x='regime', y='avg_pnl', ax=ax4, palette='magma')
        ax4.set_title('Average Profit/Loss per Trade'); ax4.set_xlabel('Specialist (Regime ID)'); ax4.set_ylabel('Avg PnL ($)')
        
        plt.savefig(self.plot_dir / "specialist_squad_report.png")
        plt.close()

    def run_all_plots(self):
        """ A single command to generate the complete intelligence dashboard. """
        if not self.data_loaded:
            get_logger(__name__).error("Plot generation skipped because data loading failed.")
            return
        
        logger = get_logger(__name__)
        logger.info("Generating complete strategic visualization dashboard...")
        self.plot_equity_curve()
        self.plot_hourly_performance()
        self.plot_daily_performance()
        self.plot_specialist_squad_report()
        logger.info("✅ All plots generated successfully.")

if __name__ == "__main__":
    plotter = HierarchicalPlotter()
    plotter.run_all_plots()