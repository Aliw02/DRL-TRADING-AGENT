# scripts/train_specialists.py
# ELITE AGENT FORGING PROTOCOL FOR HIERARCHICAL DEPLOYMENT

import pandas as pd
import joblib
import os
import sys
import gc
import json

# --- Add project root to path ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.preprocessing import RobustScaler
from config import paths
from config.init import Config
from utils.logger import setup_logging, get_logger
# --- CRITICAL IMPORT: We reuse our battle-tested single-agent training function ---
from scripts.train_agent import train_one_segment

def create_regime_datasets(full_df: pd.DataFrame, regime_classifier, regime_scaler) -> dict:
    """
    Analyzes the full dataset, assigns a regime to each timestep using the
    master classifier, and then partitions the data into dedicated datasets,
    one for each identified market regime.
    """
    logger = get_logger(__name__)
    logger.info("Partitioning master dataset by market regime...")

    # These are the features the regime classifier was trained on
    regime_features = ['adx', 'bb_width', 'roc_norm', 'rsi_x_adx', 'atr']
    X_raw = full_df[regime_features].dropna()
    X_scaled = regime_scaler.transform(X_raw)

    # Predict the dominant regime for each historical timestep
    regime_labels = regime_classifier.predict(X_scaled)
    
    # Align the dataframe with the predictions and assign the regime label
    df_aligned = full_df.loc[X_raw.index].copy()
    df_aligned['regime'] = regime_labels

    regime_datasets = {}
    for regime_id in range(regime_classifier.n_components):
        regime_df = df_aligned[df_aligned['regime'] == regime_id]
        # Drop the now-redundant regime column from the specialist's training data
        regime_datasets[regime_id] = regime_df.drop(columns=['regime'])
        logger.info(f"Regime {regime_id} dataset created with {len(regime_df)} samples.")

    return regime_datasets

def run_specialist_training_pipeline(config_path: str):
    """
    Main pipeline to forge a dedicated specialist agent for each market regime.
    This version includes the CRITICAL step of creating and saving a
    dedicated scaler for each specialist.
    """
    setup_logging()
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info(" HIERARCHICAL REINFORCEMENT LEARNING: SPECIALIST FORGING PROTOCOL ")
    logger.info("="*80)

    try:
        # --- 1. Load Master Artifacts & Configuration ---
        agent_config = Config(config_path=config_path)
        full_df = pd.read_parquet(paths.FINAL_ENRICHED_DATA_FILE)
        regime_model = joblib.load(paths.FINAL_MODEL_DIR / "regime_gmm_model.joblib")
        regime_scaler = joblib.load(paths.FINAL_MODEL_DIR / "regime_robust_scaler.joblib")
        
        min_samples_requirement = agent_config.get('specialist_training.min_samples_per_regime', 5000)
        logger.info(f"Strategic Directive: Minimum samples per regime set to {min_samples_requirement}.")

        # --- 2. Create Regime-Specific Datasets ---
        regime_datasets = create_regime_datasets(full_df, regime_model, regime_scaler)

        # --- 3. Train One Specialist Agent Per Regime ---
        num_regimes = regime_model.n_components
        for regime_id in range(num_regimes):
            logger.info("\n" + "="*60)
            logger.info(f" COMMENCING TRAINING FOR SPECIALIST AGENT: REGIME {regime_id} ")
            logger.info("="*60)

            specialist_save_dir = paths.FINAL_MODEL_DIR / f"specialist_regime_{regime_id}"
            os.makedirs(specialist_save_dir, exist_ok=True) # Ensure directory exists
            
            if (specialist_save_dir / "models/best_model.zip").exists():
                logger.info(f"Specialist for Regime {regime_id} already trained. Skipping.")
                continue

            regime_df = regime_datasets[regime_id]
            if len(regime_df) < min_samples_requirement:
                logger.warning(f"Skipping Regime {regime_id} due to insufficient data.")
                continue

            # --- CRITICAL FIX: DATA PREPARATION AND SCALER SAVING ---
            # 1. Partition the specialist's data into training and evaluation sets
            train_size = int(len(regime_df) * 0.85)
            train_df = regime_df.iloc[:train_size].copy() # Use .copy() to avoid SettingWithCopyWarning
            eval_df = regime_df.iloc[train_size:].copy()

            # 2. Define feature columns for the specialist
            feature_cols = [c for c in train_df.columns if c not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
            
            # 3. Create and fit a dedicated scaler ONLY on the specialist's training data
            specialist_scaler = RobustScaler()
            train_df[feature_cols] = specialist_scaler.fit_transform(train_df[feature_cols])
            
            # 4. Transform the evaluation data using the SAME scaler
            eval_df[feature_cols] = specialist_scaler.transform(eval_df[feature_cols])

            # 5. Save the specialist's unique scaler (their data fingerprint)
            scaler_path = specialist_save_dir / f"specialist_scaler_regime_{regime_id}.joblib"
            joblib.dump(specialist_scaler, scaler_path)
            logger.info(f"✅ Specialist scaler for Regime {regime_id} saved to: {scaler_path}")
            # ----------------------------------------------------------------

            # Deploy the standardized training protocol on the NOW-SCALED data
            train_one_segment(
                train_df=train_df,
                eval_df=eval_df,
                save_path_prefix=str(specialist_save_dir),
                config=agent_config
            )
            gc.collect()

        logger.info("\n" + "="*80)
        logger.info("✅ ALL SPECIALIST AGENT TRAINING PROTOCOLS COMPLETE.")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"A critical failure occurred during specialist training: {e}", exc_info=True)
        raise