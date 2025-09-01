# scripts/backtest_agent.py (UPGRADED FOR LIVE TRADING SIMULATION)

import pandas as pd
import joblib
import os
import sys
import numpy as np

# --- Add project path to allow imports from other directories ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import SAC
from config import paths
from envs.trading_env import TradingEnv
from utils.data_transformation import DataTransformer
from utils.logger import setup_logging, get_logger
from utils.metrics import calculate_performance_metrics

def run_backtest():
    """
    Runs a professional backtest simulating live trading, tracking and
    saving all trades based on ATR-based SL/TP and a dynamic CE-based TP.
    """
    setup_logging()
    logger = get_logger(__name__)

    logger.info("="*80)
    logger.info("STARTING LIVE TRADING SIMULATION BACKTEST")
    logger.info("="*80)

    try:
        # --- 1. Load all necessary production artifacts ---
        logger.info("Loading production agent, scalers, and regime model...")
        agent_model = SAC.load(str(paths.FINAL_MODEL_PATH))
        agent_scaler = joblib.load(str(paths.FINAL_SCALER_PATH))
        regime_model = joblib.load(paths.FINAL_MODEL_DIR / "regime_gmm_model.joblib")
        regime_scaler = joblib.load(paths.FINAL_MODEL_DIR / "regime_robust_scaler.joblib")
        logger.info("All production artifacts loaded successfully.")

        # --- 2. Load and apply full data pipeline to UNSEEN backtest data ---
        logger.info(f"Loading and processing unseen backtest data from: {paths.BACKTEST_DATA_FILE}")
        transformer = DataTransformer()
        backtest_df = transformer.load_and_preprocess_data(file_path=str(paths.BACKTEST_DATA_FILE))

        # Check for required indicators
        if 'atr' not in backtest_df.columns or 'bullish_flip' not in backtest_df.columns or 'bearish_flip' not in backtest_df.columns:
            raise ValueError("Required indicators (ATR, bullish_flip, bearish_flip) are missing from the backtest data. Please ensure they are calculated in `utils/custom_indicators.py`.")
        
        regime_features = ['adx', 'bb_width', 'roc_norm', 'rsi_x_adx', 'atr']
        X_raw = backtest_df[regime_features].ffill().bfill()
        X_scaled = regime_scaler.transform(X_raw)
        regime_probabilities = regime_model.predict_proba(X_scaled)
        prob_cols = [f'regime_prob_{i}' for i in range(regime_model.n_components)]
        prob_df = pd.DataFrame(regime_probabilities, index=backtest_df.index, columns=prob_cols)
        enriched_backtest_df = backtest_df.join(prob_df)

        # --- 3. Scale the final enriched features with the AGENT's specific scaler ---
        logger.info("Scaling final feature set for the agent...")
        feature_cols = [col for col in enriched_backtest_df.columns if col not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
        scaled_features = agent_scaler.transform(enriched_backtest_df[feature_cols])
        final_backtest_df = pd.DataFrame(scaled_features, columns=feature_cols, index=enriched_backtest_df.index)
        
        # Add original OHLCV, ATR and Flip features back for trading logic
        final_backtest_df = final_backtest_df.join(backtest_df[['open', 'high', 'low', 'close', 'atr', 'bullish_flip', 'bearish_flip', 'volume']])
        final_backtest_df.dropna(inplace=True)
        final_backtest_df.reset_index(inplace=True)

        logger.info(f"Prepared {len(final_backtest_df)} data points for backtesting.")

        # --- 4. Simulation with Trade Management ---
        # Constants for trade management
        initial_balance = 10000
        commission_per_lot = 5  # Example: $5 per standard lot
        lot_size = 0.01  # Mini lot
        
        # New configurable parameters for TP/SL
        USE_DYNAMIC_TP = True # Use Chandelier Exit flips for TP
        ATR_TP_MUL = 3.0      # ATR multiplier for fixed TP
        ATR_SL_MUL = 4.0      # ATR multiplier for SL

        equity_curve = [initial_balance]
        trades = []
        open_position = None
        pnl = 0.0

        for i in range(len(final_backtest_df) - 1):
            current_bar = final_backtest_df.iloc[i]
            next_bar = final_backtest_df.iloc[i+1]
            obs = final_backtest_df.loc[i:i].drop(columns=['open', 'high', 'low', 'close', 'atr', 'bullish_flip', 'bearish_flip', 'volume'])
            
            # Get action from the model
            action, _ = agent_model.predict(obs, deterministic=True)
            action = int(np.round(action[0])) # Convert continuous action to discrete (0, 1, 2)

            # --- Check for open position and manage it ---
            if open_position:
                # Check Stop Loss
                if open_position['type'] == 'BUY' and next_bar['low'] <= open_position['stop_loss']:
                    close_price = open_position['stop_loss']
                    pnl = (close_price - open_position['entry_price']) * (lot_size * 100000)
                    close_time = next_bar['timestamp']
                    exit_reason = 'SL'
                    open_position = None
                elif open_position['type'] == 'SELL' and next_bar['high'] >= open_position['stop_loss']:
                    close_price = open_position['stop_loss']
                    pnl = (open_position['entry_price'] - close_price) * (lot_size * 100000)
                    close_time = next_bar['timestamp']
                    exit_reason = 'SL'
                    open_position = None
                
                # Check Take Profit
                elif USE_DYNAMIC_TP and ((open_position['type'] == 'BUY' and current_bar['bearish_flip'] == 1) or \
                                         (open_position['type'] == 'SELL' and current_bar['bullish_flip'] == 1)):
                    close_price = next_bar['close']
                    pnl = (close_price - open_position['entry_price']) * (lot_size * 100000) if open_position['type'] == 'BUY' else (open_position['entry_price'] - close_price) * (lot_size * 100000)
                    close_time = next_bar['timestamp']
                    exit_reason = 'Dynamic TP'
                    open_position = None
                elif not USE_DYNAMIC_TP and ((open_position['type'] == 'BUY' and next_bar['high'] >= open_position['take_profit']) or \
                                             (open_position['type'] == 'SELL' and next_bar['low'] <= open_position['take_profit'])):
                    close_price = open_position['take_profit']
                    pnl = (close_price - open_position['entry_price']) * (lot_size * 100000) if open_position['type'] == 'BUY' else (open_position['entry_price'] - close_price) * (lot_size * 100000)
                    close_time = next_bar['timestamp']
                    exit_reason = 'Fixed TP'
                    open_position = None
                
                # Check for action to close position
                elif (open_position['type'] == 'BUY' and action == 2) or (open_position['type'] == 'SELL' and action == 1):
                    close_price = next_bar['close']
                    pnl = (close_price - open_position['entry_price']) * (lot_size * 100000) if open_position['type'] == 'BUY' else (open_position['entry_price'] - close_price) * (lot_size * 100000)
                    close_time = next_bar['timestamp']
                    exit_reason = 'Model Flip'
                    open_position = None

                if not open_position:
                    commission = (lot_size * 2) * commission_per_lot
                    net_pnl = pnl - commission
                    trades.append({
                        'entry_time': open_position['entry_time'],
                        'exit_time': close_time,
                        'type': open_position['type'],
                        'entry_price': open_position['entry_price'],
                        'exit_price': close_price,
                        'gross_profit': pnl,
                        'commission': commission,
                        'net_profit': net_pnl,
                        'exit_reason': exit_reason
                    })
                    initial_balance += net_pnl
            
            # --- Open a new position if none exists and model recommends ---
            if not open_position:
                if action == 1: # BUY
                    entry_price = next_bar['open']
                    atr = next_bar['atr']
                    stop_loss = entry_price - (atr * ATR_SL_MUL)
                    take_profit = entry_price + (atr * ATR_TP_MUL)
                    open_position = {
                        'entry_time': next_bar['timestamp'],
                        'type': 'BUY',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                elif action == 2: # SELL
                    entry_price = next_bar['open']
                    atr = next_bar['atr']
                    stop_loss = entry_price + (atr * ATR_SL_MUL)
                    take_profit = entry_price - (atr * ATR_TP_MUL)
                    open_position = {
                        'entry_time': next_bar['timestamp'],
                        'type': 'SELL',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
            
            # Update equity curve
            current_equity = initial_balance
            if open_position:
                current_price = final_backtest_df.iloc[i+1]['close']
                unrealized_pnl = (current_price - open_position['entry_price']) * (lot_size * 100000) if open_position['type'] == 'BUY' else (open_position['entry_price'] - current_price) * (lot_size * 100000)
                current_equity += unrealized_pnl
            
            equity_curve.append(current_equity)

        # --- 5. Save and display results ---
        trades_df = pd.DataFrame(trades)
        equity_series = pd.Series(equity_curve)
        
        trades_df.to_csv(paths.RESULTS_DIR / "final_backtest_trades.csv", index=False)
        equity_series.to_csv(paths.RESULTS_DIR / "final_backtest_equity.csv", index=False)
        
        logger.info(f"Backtest equity curve saved to: {paths.RESULTS_DIR / 'final_backtest_equity.csv'}")
        logger.info(f"Backtest trades log saved to: {paths.RESULTS_DIR / 'final_backtest_trades.csv'}")

        performance_stats = calculate_performance_metrics(equity_series)

        print("\n" + "="*50)
        print("      PROFESSIONAL BACKTESTING REPORT")
        print("="*50)
        print(f"Total Trades:            {len(trades_df)}")
        if not trades_df.empty:
            profitable_trades = trades_df[trades_df['net_profit'] > 0]
            win_rate = len(profitable_trades) / len(trades_df)
            print(f"Win Rate:                {win_rate * 100:.2f}%")
            print(f"Profit Factor:           {(profitable_trades['net_profit'].sum() / abs(trades_df[trades_df['net_profit'] < 0]['net_profit'].sum())):.2f}")
        
        print("\nRISK-ADJUSTED METRICS:")
        print(f"Initial Portfolio Value: ${equity_series.iloc[0]:,.2f}")
        print(f"Final Portfolio Value:   ${equity_series.iloc[-1]:,.2f}")
        print(f"Total Return:            {((equity_series.iloc[-1] / equity_series.iloc[0]) - 1) * 100:.2f}%")
        print(f"Annualized Sharpe Ratio: {performance_stats['sharpe_ratio']:.2f}")
        print(f"Annualized Sortino Ratio: {performance_stats['sortino_ratio']:.2f}")
        print(f"Calmar Ratio:            {performance_stats['calmar_ratio']:.2f}")
        print(f"Maximum Drawdown:        {performance_stats['max_drawdown'] * 100:.2f}%")
        print("="*50)

    except FileNotFoundError as e:
        logger.error(f"A required file was not found: {e}")
        logger.error("Please ensure the full training pipeline has been run successfully before backtesting.")
    except Exception as e:
        logger.error(f"An error occurred during backtesting: {e}", exc_info=True)

if __name__ == "__main__":
    run_backtest()