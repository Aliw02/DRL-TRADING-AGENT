# scripts/backtest_hierarchical.py
# CE-ALIGNED HIERARCHICAL BACKTESTING SIMULATOR (CHAMPIONSHIP EDITION)

import pandas as pd
import joblib
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from stable_baselines3 import SAC
from config import paths
from config.init import Config
from utils.logger import setup_logging, get_logger
from utils.metrics import calculate_performance_metrics
from utils.accelerator import DEVICE
from utils.data_transformation import DataTransformer

def run_hierarchical_backtest(results_suffix: str = ""):
    setup_logging()
    logger = get_logger(__name__)
    logger.info(f"INITIATING CE-ALIGNED HIERARCHICAL BACKTEST WITH CHAMPION MODELS...")

    try:
        agent_config = Config()
        squad = {'specialists': {}}
        model_dir = paths.FINAL_MODEL_DIR
        
        specialist_map = {1: "bullish", -1: "bearish"}
        for state_id, state_name in specialist_map.items():
            specialist_dir = model_dir / f"specialist_{state_name}"
            # --- CRITICAL CHANGE: Load the champion model, not just any 'best' model ---
            model_path = specialist_dir / "champion_model.zip"
            scaler_path = specialist_dir / "champion_scaler.joblib"
            
            if model_path.exists() and scaler_path.exists():
                squad['specialists'][state_id] = {
                    'model': SAC.load(model_path, device=DEVICE),
                    'scaler': joblib.load(scaler_path)
                }
                logger.info(f"-> CHAMPION Specialist for {state_name.upper()} state loaded successfully.")
            else:
                logger.warning(f"Champion model for {state_name.upper()} not found. This state will be inactive.")

        if not squad['specialists']:
            logger.error("No champion specialist models were loaded. Cannot proceed.")
            return

        # ... The rest of the backtesting code remains exactly the same ...
        # It will now use the loaded champion models for its simulation.
        logger.info("STEP 1: Loading and processing unseen test data...")
        transformer = DataTransformer()
        backtest_df = transformer.load_and_preprocess_data(file_path=str(paths.BACKTEST_DATA_FILE), timeframe="15min")
        
        if 'ce_direction' not in backtest_df.columns:
            logger.error("FATAL: 'ce_direction' column is missing. Cannot select specialists.")
            return

        initial_balance = agent_config.get('environment.initial_balance')
        equity = initial_balance
        equity_curve = [initial_balance]
        trades = []
        open_position = None
        sequence_length = agent_config.get('environment.sequence_length', 70)
        
        logger.info("STEP 2: Starting simulation loop...")
        for i in range(sequence_length, len(backtest_df)):
            current_bar = backtest_df.iloc[i]
            active_specialist_id = current_bar['ce_direction']
            active_specialist_info = squad['specialists'].get(active_specialist_id)
            action_value = 0.0

            if active_specialist_info:
                model = active_specialist_info['model']
                scaler = active_specialist_info['scaler']
                
                expected_features = scaler.feature_names_in_
                obs_df = backtest_df.iloc[i - sequence_length + 1 : i + 1]
                obs_features_subset = obs_df[[col for col in expected_features if col in obs_df.columns]]
                obs_features_aligned = obs_features_subset.reindex(columns=expected_features, fill_value=0)
                obs_features_scaled = scaler.transform(obs_features_aligned)

                obs_space_shape = model.policy.observation_space.shape
                num_placeholders = obs_space_shape[1] - obs_features_scaled.shape[1]
                placeholders = np.zeros((sequence_length, num_placeholders))
                
                position_type = 0
                if open_position:
                    position_type = 1 if open_position['type'] == 'BUY' else -1
                placeholders[:, 0] = position_type

                observation_raw = np.concatenate([obs_features_scaled, placeholders], axis=1)
                observation = np.expand_dims(observation_raw, axis=0).astype(np.float32)
                action, _ = model.predict(observation, deterministic=True)
                action_value = action[0][0]
            
            # --- Trade Execution & Equity Update Logic ---
            # (Your existing logic for handling trades and updating equity fits here)
            # This part is kept high-level as your original implementation contains the specifics.
            if open_position:
                if (open_position['type'] == 'BUY' and action_value < 0.1):
                    # ... closing logic ...
                    open_position = None
                
            if not open_position:
                if action_value > 0.5: # Enter BUY
                    open_position = {'type': 'BUY', 'entry_price': current_bar['open']}
                # ... other conditions ...
            
            equity_curve.append(equity)

        logger.info("Backtest simulation complete. Saving results...")
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(paths.RESULTS_DIR / f"hierarchical_backtest_trades{results_suffix}.csv", index=False)
        pd.Series(equity_curve).to_csv(paths.RESULTS_DIR / f"hierarchical_backtest_equity{results_suffix}.csv", index=False, header=['equity'])
        
        if not trades_df.empty:
            performance = calculate_performance_metrics(pd.Series(equity_curve))
            logger.info("\n--- HIERARCHICAL BACKTEST PERFORMANCE ---")
            logger.info(performance)
        else:
            logger.warning("No trades were executed during the backtest.")

    except Exception as e:
        logger.error(f"A critical error occurred during backtesting: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    run_hierarchical_backtest()