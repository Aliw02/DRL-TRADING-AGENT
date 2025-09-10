# scripts/plot_results.py
# FINAL, INDEX-AWARE, AND ROBUST VISUALIZATION DASHBOARD

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
from utils.logger import get_logger, setup_logging

class HierarchicalPlotter:
    """
    A comprehensive visualization suite for analyzing the performance of the
    hierarchical squad of specialist agents.
    """
    def __init__(self, results_suffix=""):
        self.plot_dir = paths.RESULTS_DIR / f"plots{results_suffix}"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.results_suffix = results_suffix
        self.data_loaded = self._load_data()

    def _load_data(self) -> bool:
        """
        Robustly loads all data files required for plotting.
        """
        logger = get_logger(__name__)
        setup_logging()
        try:
            equity_path = paths.RESULTS_DIR / f"hierarchical_backtest_equity{self.results_suffix}.csv"
            trades_path = paths.RESULTS_DIR / f"hierarchical_backtest_trades{self.results_suffix}.csv"
            
            self.equity_df = pd.read_csv(equity_path)
            
            if os.path.exists(trades_path) and os.path.getsize(trades_path) > 5:
                self.trades_df = pd.read_csv(trades_path, parse_dates=['entry_time', 'exit_time'])
            else:
                logger.warning("Trades file is empty or not found. No trade-specific plots will be generated.")
                self.trades_df = pd.DataFrame(columns=['entry_time', 'exit_time', 'net_profit', 'regime'])

            # --- التحسين الرئيسي: إعادة الفهرس ليصبح عموداً ---
            # We need a corresponding price data file to get the timestamps.
            # Assuming the backtest was run on the full enriched data.
            price_data_df = pd.read_parquet(paths.PROCESSED_DATA_FILE).reset_index()
            
            # Align the equity curve with the timestamps from the price data
            if 'timestamp' in price_data_df.columns and len(self.equity_df) <= len(price_data_df):
                self.equity_df['timestamp'] = price_data_df['timestamp'].iloc[-len(self.equity_df):].values
            else:
                logger.warning("Could not align timestamps for the equity curve plot.")
                # Create a placeholder index if alignment fails
                self.equity_df['timestamp'] = pd.to_datetime(self.equity_df.index)


            logger.info("All necessary data for plotting is loaded and validated.")
            return True
        except FileNotFoundError as e:
            logger.error(f"A required file for plotting was not found: {e}. Plotting will be skipped.")
            return False
        except Exception as e:
            logger.error(f"A critical error occurred during data loading for plots: {e}", exc_info=True)
            return False

    def plot_equity_curve(self):
        """ Plots the main equity curve and visualizes drawdown periods. """
        if 'timestamp' not in self.equity_df.columns:
            logger.error("Timestamp column is missing in equity data, cannot plot equity curve.")
            return
            
        logger = get_logger(__name__)
        logger.info("Generating main equity curve plot...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Ensure timestamp is in datetime format for plotting
        self.equity_df['timestamp'] = pd.to_datetime(self.equity_df['timestamp'])

        ax.plot(self.equity_df['timestamp'], self.equity_df['equity'], label='Hierarchical Squad Equity', color='navy', linewidth=2.5)
        
        high_water_mark = self.equity_df['equity'].cummax()
        ax.fill_between(self.equity_df['timestamp'], self.equity_df['equity'], high_water_mark, 
                        where=(self.equity_df['equity'] < high_water_mark),
                        color='red', alpha=0.2, interpolate=True, label='Drawdown Period')

        ax.set_title(f'Squad Performance: Equity Curve{self.results_suffix}', fontsize=18, weight='bold')
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

    def plot_specialist_squad_report(self):
        """
        Analyzes and visualizes the performance of each individual specialist agent.
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
        self.plot_specialist_squad_report()
        logger.info("✅ All plots generated successfully.")

if __name__ == "__main__":
    plotter = HierarchicalPlotter()
    plotter.run_all_plots()