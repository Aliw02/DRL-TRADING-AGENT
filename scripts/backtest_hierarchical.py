# scripts/backtest_hierarchical.py
# FINAL, INTELLIGENCE-AWARE HIERARCHICAL SQUAD COMBAT SIMULATOR

import pandas as pd
import joblib
import os
import sys
import numpy as np
import json

# --- Add project root to path for robust imports ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from stable_baselines3 import SAC
from config import paths
from config.init import Config
from utils.logger import setup_logging, get_logger
from utils.metrics import calculate_performance_metrics
from utils.accelerator import DEVICE

def run_hierarchical_backtest(backtest_df: pd.DataFrame = None, results_suffix: str = ""):
    """
    Executes a full backtest using the hierarchical squad. It now logs the
    responsible regime for every trade, enabling advanced performance analysis.
    """
    setup_logging()
    logger = get_logger(__name__)
    logger.info(f"INITIATING HIERARCHICAL BACKTEST (Results suffix: '{results_suffix}')...")

    try:
        # --- 1. Load the Entire Specialist Squad and Command Unit ---
        agent_config = Config()
        squad = {}
        logger.info("Loading Command Unit and Specialist Squad...")
        squad['classifier'] = joblib.load(paths.FINAL_MODEL_DIR / "regime_gmm_model.joblib")
        squad['scaler'] = joblib.load(paths.FINAL_MODEL_DIR / "regime_robust_scaler.joblib")
        
        with open(paths.FINAL_MODEL_DIR / "regime_config.json", 'r') as f:
            num_regimes = json.load(f)['optimal_n_clusters']
        
        squad['specialists'] = {}
        for i in range(num_regimes):
            model_path = paths.FINAL_MODEL_DIR / f"specialist_regime_{i}/models/best_model.zip"
            if os.path.exists(model_path):
                squad['specialists'][i] = SAC.load(model_path, device=DEVICE)

        # --- 2. Prepare Unseen Data ---
        if backtest_df is None:
            logger.info(f"Loading standard unseen data for backtest...")
            backtest_df = pd.read_parquet(paths.FINAL_ENRICHED_DATA_FILE).tail(100000)
        else:
            logger.info("Using custom data for stress testing scenario...")
        
        # --- 3. Simulation Core ---
        initial_balance = agent_config.get('environment.initial_balance')
        equity = initial_balance
        equity_curve = [initial_balance]
        trades = []
        open_position = None
        sequence_length = agent_config.get('environment.sequence_length')
        
        regime_features = ['adx', 'bb_width', 'roc_norm', 'rsi_x_adx', 'atr']
        specialist_feature_cols = [col for col in backtest_df.columns if 'regime_prob' not in col and col not in ['open', 'high', 'low', 'close', 'time', 'timestamp', 'regime']]

        logger.info("Starting hierarchical simulation loop...")
        for i in range(sequence_length, len(backtest_df)):
            current_bar = backtest_df.iloc[i]

            # --- COMMAND STEP: Determine current market regime ---
            current_regime_features = squad['scaler'].transform(current_bar[regime_features].values.reshape(1, -1))
            current_regime = squad['classifier'].predict(current_regime_features)[0]

            # --- TACTICAL STEP: Activate specialist and get action ---
            active_specialist = squad['specialists'].get(current_regime)
            action_value = 0.0

            if active_specialist:
                obs_df = backtest_df.iloc[i - sequence_length + 1 : i + 1]
                obs_features = obs_df[specialist_feature_cols].values
                pos_size = 1.0 if open_position else 0.0
                pos_feature = np.full((sequence_length, 1), pos_size, dtype=np.float32)
                observation = np.expand_dims(np.concatenate([obs_features, pos_feature], axis=1), axis=0)
                action, _ = active_specialist.predict(observation, deterministic=True)
                action_value = action[0][0]

            # --- EXECUTION STEP with INTELLIGENCE LOGGING ---
            BUY_THRESHOLD, SELL_THRESHOLD = 0.5, -0.5
            if open_position and action_value < SELL_THRESHOLD:
                pnl = current_bar['open'] - open_position['entry_price']
                equity += pnl
                trades.append({
                    'entry_time': open_position['entry_time'],
                    'exit_time': current_bar['timestamp'],
                    'type': open_position['type'],
                    'entry_price': open_position['entry_price'],
                    'exit_price': current_bar['open'],
                    'net_profit': pnl,
                    # --- CRITICAL INTELLIGENCE DATA POINT ---
                    'regime': open_position['regime_at_entry']
                })
                open_position = None

            if not open_position and action_value > BUY_THRESHOLD:
                open_position = {
                    'type': 'BUY',
                    'entry_price': current_bar['open'],
                    'entry_time': current_bar['timestamp'],
                    # --- CRITICAL INTELLIGENCE DATA POINT ---
                    'regime_at_entry': current_regime
                }
            
            # Update Equity Curve
            current_equity = equity
            if open_position:
                current_equity += (current_bar['close'] - open_position['entry_price'])
            equity_curve.append(current_equity)

        # --- 4. Save and Display Final Results ---
        logger.info("Hierarchical backtest complete. Saving intelligence logs...")
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(paths.RESULTS_DIR / f"hierarchical_backtest_trades{results_suffix}.csv", index=False)
        pd.Series(equity_curve).to_csv(paths.RESULTS_DIR / f"hierarchical_backtest_equity{results_suffix}.csv", index=False, header=['equity'])
        
        if not trades_df.empty:
            performance = calculate_performance_metrics(pd.Series(equity_curve))
            logger.info("\n--- HIERARCHICAL BACKTEST PERFORMANCE ---")
            logger.info(performance)
            logger.info("------------------------------------")
        else:
            logger.warning("No trades were executed during the backtest.")

    except Exception as e:
        logger.error(f"A critical error occurred during hierarchical backtesting: {e}", exc_info=True)
        raise