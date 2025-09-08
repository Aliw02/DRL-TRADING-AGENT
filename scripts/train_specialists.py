# scripts/train_specialists.py
# ELITE AGENT FORGING PROTOCOL WITH ADAPTIVE WALK-FORWARD VALIDATION

import pandas as pd
import joblib
import os
import sys
import gc
from sklearn.preprocessing import RobustScaler
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC

# --- Add project root to path ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import paths
from config.init import Config
from utils.logger import setup_logging, get_logger
from utils.accelerator import DEVICE
from scripts.train_agent import train_one_segment
from envs.trading_env import TradingEnv
from stable_baselines3.common.env_util import make_vec_env

def create_regime_datasets(full_df: pd.DataFrame, regime_classifier, regime_scaler) -> dict:
    """
    Analyzes the full dataset, assigns a regime to each timestep, and partitions the data.
    """
    logger = get_logger(__name__)
    logger.info("Partitioning master dataset by market regime...")
    regime_features = ['adx', 'bb_width', 'roc_norm', 'rsi_x_adx', 'atr']
    X_raw = full_df[regime_features].dropna()
    X_scaled = regime_scaler.transform(X_raw)
    regime_labels = regime_classifier.predict(X_scaled)
    df_aligned = full_df.loc[X_raw.index].copy()
    df_aligned['regime'] = regime_labels
    regime_datasets = {}
    for regime_id in range(regime_classifier.n_components):
        regime_df = df_aligned[df_aligned['regime'] == regime_id]
        regime_datasets[regime_id] = regime_df.drop(columns=['regime'])
        logger.info(f"Regime {regime_id} dataset created with {len(regime_df)} samples.")
    return regime_datasets

def run_specialist_walk_forward(regime_df: pd.DataFrame, specialist_save_dir: str, config: Config, regime_id: int):
    """
    Applies a robust walk-forward training with parameters suitable for smaller, specialized datasets.
    """
    logger = get_logger(__name__)
    
    # --- CRITICAL FIX: Use dedicated, more flexible parameters for specialists ---
    # Instead of inheriting the main config's demanding values, we define
    # a more appropriate set of parameters here.
    n_splits = 3
    min_train_size = 8000
    test_size = 2500
    holdout_size = 2500
    
    # Calculate the actual data requirement for this specialist
    total_required_samples = min_train_size + (n_splits * test_size) + holdout_size
    logger.info(f"Specialist WF Requirement: {total_required_samples} samples.")
    
    if len(regime_df) < total_required_samples:
        logger.warning(f"Regime {regime_id} has {len(regime_df)} samples, but requires {total_required_samples}. Skipping.")
        return

    feature_cols = [c for c in regime_df.columns if c not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
    
    training_data_end = len(regime_df) - holdout_size
    training_df = regime_df.iloc[:training_data_end]
    holdout_df = regime_df.iloc[training_data_end:]

    for i in range(n_splits):
        split_num = i + 1
        logger.info(f"\n--- Specialist {regime_id}: Processing Walk-Forward Split {split_num}/{n_splits} ---")
        
        split_save_path = os.path.join(specialist_save_dir, f"split_{split_num}")
        if os.path.exists(os.path.join(split_save_path, "models/best_model.zip")):
            logger.info(f"Split {split_num} for specialist {regime_id} already trained. Skipping.")
            continue

        train_end = min_train_size + i * test_size
        test_end = train_end + test_size
        if test_end > len(training_df):
            logger.warning("Not enough data for split. Halting process for this specialist.")
            break

        train_df_split = training_df.iloc[:train_end].copy()
        test_df_split = training_df.iloc[train_end:test_end].copy()

        scaler = RobustScaler()
        train_df_split.loc[:, feature_cols] = scaler.fit_transform(train_df_split[feature_cols])
        test_df_split.loc[:, feature_cols] = scaler.transform(test_df_split[feature_cols])
        
        os.makedirs(split_save_path, exist_ok=True)
        joblib.dump(scaler, os.path.join(split_save_path, f"scaler_split_{split_num}.joblib"))

        train_one_segment(train_df_split, test_df_split, split_save_path, config)
        gc.collect()

    logger.info(f"--- Specialist {regime_id}: Finding champion model ---")
    best_model_path, best_reward = None, -float('inf')
    for i in range(1, n_splits + 1):
        model_path = os.path.join(specialist_save_dir, f"split_{i}/models/best_model.zip")
        scaler_path = os.path.join(specialist_save_dir, f"split_{i}/scaler_split_{i}.joblib")
        if not os.path.exists(model_path): continue
        
        model = SAC.load(model_path, device=DEVICE)
        scaler = joblib.load(scaler_path)
        
        holdout_df_scaled = holdout_df.copy()
        holdout_df_scaled.loc[:, feature_cols] = scaler.transform(holdout_df_scaled[feature_cols])
        
        eval_env = make_vec_env(lambda: TradingEnv(holdout_df_scaled), n_envs=1)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=1, deterministic=True)
        logger.info(f"Split {i} model reward on specialist holdout set: {mean_reward:.2f}")
        
        if mean_reward > best_reward:
            best_reward, best_model_path = mean_reward, model_path
            # =================== FIX 1 START ===================
            # Use os.path.join for string-based path concatenation
            final_scaler_path = os.path.join(specialist_save_dir, f"specialist_scaler_regime_{regime_id}.joblib")
            joblib.dump(scaler, final_scaler_path)
            # =================== FIX 1 END =====================
            logger.info(f"🏆 New champion for Specialist {regime_id}! Split {i} is best.")
        del model, eval_env

    if best_model_path:
        # =================== FIX 2 START ===================
        # Use os.path.join for string-based path concatenation
        final_model_dir = os.path.join(specialist_save_dir, "models")
        final_model_path = os.path.join(final_model_dir, "best_model.zip")
        os.makedirs(final_model_dir, exist_ok=True)
        # =================== FIX 2 END =====================
        
        # Use os.rename for moving the file, it's more atomic
        os.rename(best_model_path, final_model_path)
        logger.info(f"✅ Champion model for Specialist {regime_id} has been deployed.")
    else:
        logger.error(f"Could not determine a champion model for Specialist {regime_id}.")

def run_specialist_training_pipeline(config_path: str):
    """
    Main pipeline to forge specialist agents using Walk-Forward Validation.
    """
    setup_logging()
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info(" HIERARCHICAL REINFORCEMENT LEARNING: SPECIALIST FORGING PROTOCOL (ADAPTIVE WF EDITION) ")
    logger.info("="*80)

    try:
        agent_config = Config(config_path=config_path)
        full_df = pd.read_parquet(paths.FINAL_ENRICHED_DATA_FILE)
        regime_model = joblib.load(paths.FINAL_MODEL_DIR / "regime_gmm_model.joblib")
        regime_scaler = joblib.load(paths.FINAL_MODEL_DIR / "regime_robust_scaler.joblib")
        
        regime_datasets = create_regime_datasets(full_df, regime_model, regime_scaler)

        num_regimes = regime_model.n_components
        for regime_id in range(num_regimes):
            logger.info("\n" + "="*60)
            logger.info(f" COMMENCING ADAPTIVE WF TRAINING FOR SPECIALIST: REGIME {regime_id} ")
            logger.info("="*60)
            
            specialist_save_dir = str(paths.FINAL_MODEL_DIR / f"specialist_regime_{regime_id}")
            if os.path.exists(os.path.join(specialist_save_dir, "models/best_model.zip")):
                logger.info(f"Champion for Specialist {regime_id} already exists. Skipping.")
                continue

            regime_df = regime_datasets.get(regime_id)
            if regime_df is None or regime_df.empty:
                logger.warning(f"No data found for Regime {regime_id}. Skipping.")
                continue

            run_specialist_walk_forward(regime_df, specialist_save_dir, agent_config, regime_id)

        logger.info("\n" + "="*80)
        logger.info("✅ ALL SPECIALIST AGENT FORGING PROTOCOLS COMPLETE.")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"A critical failure occurred during specialist training: {e}", exc_info=True)
        raise