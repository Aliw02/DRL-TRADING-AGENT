# scripts/backtest_hierarchical.py
# HIERARCHICAL SQUAD COMBAT SIMULATOR

import pandas as pd
import joblib
import os
import sys
import numpy as np
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import SAC
from config import paths
from config.init import Config
from utils.logger import setup_logging, get_logger
from utils.metrics import calculate_performance_metrics
from utils.accelerator import DEVICE

def run_hierarchical_backtest():
    """
    Executes a full backtest simulation using the hierarchical squad of specialist agents.
    It simulates the entire command chain: classify, delegate, execute.
    """
    setup_logging()
    logger = get_logger(__name__)
    logger.info("="*80); logger.info("INITIATING HIERARCHICAL SQUAD BACKTEST"); logger.info("="*80)

    try:
        # --- 1. Load the Entire Specialist Squad and Command Unit ---
        agent_config = Config()
        squad = {}
        logger.info("Loading Command Unit (Regime Classifier and Scaler)...")
        squad['classifier'] = joblib.load(paths.FINAL_MODEL_DIR / "regime_gmm_model.joblib")
        squad['scaler'] = joblib.load(paths.FINAL_MODEL_DIR / "regime_robust_scaler.joblib")
        
        with open(paths.FINAL_MODEL_DIR / "regime_config.json", 'r') as f:
            num_regimes = json.load(f)['optimal_n_clusters']
        
        logger.info(f"Loading {num_regimes} specialist agents...")
        squad['specialists'] = {}
        for i in range(num_regimes):
            model_path = paths.FINAL_MODEL_DIR / f"specialist_regime_{i}/models/best_model.zip"
            if os.path.exists(model_path):
                squad['specialists'][i] = SAC.load(model_path, device=DEVICE)

        # --- 2. Prepare Unseen Data ---
        logger.info(f"Loading and processing unseen data from: {paths.BACKTEST_DATA_FILE}")
        # Using a large, recent chunk of the main dataset for a robust backtest
        backtest_df = pd.read_parquet(paths.FINAL_ENRICHED_DATA_FILE).tail(100000) 
        
        # --- 3. Simulation Core ---
        initial_balance = agent_config.get('environment.initial_balance')
        equity = initial_balance
        equity_curve = [initial_balance]
        trades = []
        open_position = None
        sequence_length = agent_config.get('environment.sequence_length')
        
        regime_features = ['adx', 'bb_width', 'roc_norm', 'rsi_x_adx', 'atr']
        # The specialist was trained on features WITHOUT the regime probabilities
        specialist_feature_cols = [col for col in backtest_df.columns if 'regime_prob' not in col and col not in ['open', 'high', 'low', 'close', 'time', 'timestamp', 'regime']]

        logger.info("Starting hierarchical simulation loop...")
        for i in range(sequence_length, len(backtest_df)):
            current_bar = backtest_df.iloc[i]

            # --- COMMAND STEP: Determine current market regime ---
            current_regime_features = squad['scaler'].transform(current_bar[regime_features].values.reshape(1, -1))
            current_regime = squad['classifier'].predict(current_regime_features)[0]

            # --- TACTICAL STEP: Activate the correct specialist ---
            active_specialist = squad['specialists'].get(current_regime)
            action_value = 0.0 # Default to a neutral action (HOLD)

            if not active_specialist:
                logger.debug(f"No specialist for Regime {current_regime}. Defaulting to HOLD.")
            else:
                # Prepare observation for the specialist
                obs_df = backtest_df.iloc[i - sequence_length + 1 : i + 1]
                obs_features = obs_df[specialist_feature_cols].values
                
                current_pos_size = 1.0 if open_position else 0.0
                position_feature = np.full((sequence_length, 1), current_pos_size, dtype=np.float32)
                
                observation = np.expand_dims(np.concatenate([obs_features, position_feature], axis=1), axis=0)
                action, _ = active_specialist.predict(observation, deterministic=True)
                action_value = action[0][0]

            # --- EXECUTION STEP: (This is a simplified trade logic for demonstration) ---
            # A full implementation would use the advanced logic from your original backtester
            BUY_THRESHOLD, SELL_THRESHOLD = 0.5, -0.5
            if not open_position and action_value > BUY_THRESHOLD:
                open_position = {'type': 'BUY', 'entry_price': current_bar['open']}
                trades.append({'time': current_bar['timestamp'], 'action': 'BUY', 'price': current_bar['open']})

            elif open_position and action_value < SELL_THRESHOLD:
                pnl = current_bar['open'] - open_position['entry_price']
                equity += pnl
                trades.append({'time': current_bar['timestamp'], 'action': 'SELL', 'price': current_bar['open']})
                open_position = None
            
            # --- Update Equity Curve ---
            current_equity = equity
            if open_position:
                current_equity += (current_bar['close'] - open_position['entry_price'])
            equity_curve.append(current_equity)

        # --- 4. Save and Display Final Results ---
        logger.info("Hierarchical backtest complete. Saving results...")
        pd.DataFrame(trades).to_csv(paths.RESULTS_DIR / "hierarchical_backtest_trades.csv", index=False)
        pd.Series(equity_curve).to_csv(paths.RESULTS_DIR / "hierarchical_backtest_equity.csv", index=False, header=['equity'])
        
        performance = calculate_performance_metrics(pd.Series(equity_curve))
        logger.info("\n--- HIERARCHICAL BACKTEST PERFORMANCE ---")
        logger.info(performance)
        logger.info("------------------------------------")

    except Exception as e:
        logger.error(f"A critical error occurred during hierarchical backtesting: {e}", exc_info=True)
        raise