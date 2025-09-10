# scripts/train_specialists.py
# FINAL VERSION: CE-ALIGNED SPECIALIST FORGING WITH DYNAMIC CONFIGURATION

import pandas as pd
import joblib
import os
import sys
import gc
import shutil
from sklearn.preprocessing import RobustScaler
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import paths
from config.init import Config
from utils.logger import setup_logging, get_logger
from utils.accelerator import DEVICE
from scripts.train_agent import train_one_segment # We reuse the core training engine
from envs.trading_env import TradingEnv
from stable_baselines3.common.env_util import make_vec_env

def create_ce_state_datasets(full_df: pd.DataFrame) -> dict:
    """
    Partitions the master dataset based on the Chandelier Exit (CE) direction
    to create specialized datasets for Bullish (1) and Bearish (-1) advisors.
    """
    logger = get_logger(__name__)
    logger.info("Partitioning master dataset by CE direction for specialist training...")
    
    datasets = {}
    bullish_df = full_df[full_df['ce_direction'] == 1].copy()
    bearish_df = full_df[full_df['ce_direction'] == -1].copy()
    
    if not bullish_df.empty:
        datasets[1] = bullish_df
        logger.info(f"Bullish dataset created with {len(bullish_df)} samples.")
    else:
        logger.warning("No data found for Bullish state (ce_direction == 1).")
        
    if not bearish_df.empty:
        datasets[-1] = bearish_df
        logger.info(f"Bearish dataset created with {len(bearish_df)} samples.")
    else:
        logger.warning("No data found for Bearish state (ce_direction == -1).")
        
    return datasets

def run_specialist_walk_forward(specialist_df: pd.DataFrame, specialist_save_dir: str, config: Config, state_name: str):
    """
    Manages the entire walk-forward training, validation, and champion selection
    process for a single specialist agent. This is the gold standard for preventing
    overfitting and ensuring model robustness.
    """
    logger = get_logger(__name__)
    
    # --- Check if a champion model already exists to skip re-training ---
    final_model_path = os.path.join(specialist_save_dir, "champion_model.zip")
    final_scaler_path = os.path.join(specialist_save_dir, "champion_scaler.joblib")
    if os.path.exists(final_model_path) and os.path.exists(final_scaler_path):
        logger.info(f"✅ Champion model for {state_name} advisor already exists. Skipping training.")
        return

    wf_config = config.get('training.walk_forward')
    n_splits = wf_config['n_splits']
    min_train_size = wf_config['min_train_size']
    test_size = wf_config['test_size']
    holdout_size = test_size

    total_required_data = min_train_size + (n_splits * test_size)
    if len(specialist_df) < total_required_data:
        logger.error(f"INSUFFICIENT DATA for {state_name} advisor. Requires {total_required_data}, but only has {len(specialist_df)}. Skipping.")
        return

    feature_cols = [c for c in specialist_df.columns if c not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
    
    training_data_end = len(specialist_df) - holdout_size
    training_df = specialist_df.iloc[:training_data_end]
    holdout_df = specialist_df.iloc[training_data_end:]

    # --- Walk-Forward Training Phase ---
    for i in range(n_splits):
        split_num = i + 1
        logger.info(f"--- {state_name} Advisor: Processing Walk-Forward Split {split_num}/{n_splits} ---")
        split_save_path = os.path.join(specialist_save_dir, f"split_{split_num}")
        
        if os.path.exists(os.path.join(split_save_path, "models/best_model.zip")):
            logger.info(f"Split {split_num} for {state_name} already trained. Skipping.")
            continue

        train_end = min_train_size + i * test_size
        eval_end = train_end + test_size
        if eval_end > len(training_df):
            logger.warning(f"Not enough data for split {split_num}. Halting process.")
            break

        train_split_df = training_df.iloc[:train_end].copy()
        eval_split_df = training_df.iloc[train_end:eval_end].copy()

        scaler = RobustScaler()
        train_split_df.loc[:, feature_cols] = scaler.fit_transform(train_split_df[feature_cols])
        eval_split_df.loc[:, feature_cols] = scaler.transform(eval_split_df[feature_cols])
        
        os.makedirs(split_save_path, exist_ok=True)
        joblib.dump(scaler, os.path.join(split_save_path, "scaler.joblib"))

        train_one_segment(train_split_df, eval_split_df, split_save_path, config)
        gc.collect()

    # --- Champion Selection Phase ---
    logger.info(f"--- {state_name} Advisor: Finding Champion Model on Unseen Holdout Data ---")
    best_model_path, best_scaler_path, best_reward = None, None, -float('inf')

    for i in range(1, n_splits + 1):
        model_path = os.path.join(specialist_save_dir, f"split_{i}/models/best_model.zip")
        scaler_path = os.path.join(specialist_save_dir, f"split_{i}/scaler.joblib")
        if not os.path.exists(model_path): continue

        model = SAC.load(model_path, device=DEVICE)
        scaler = joblib.load(scaler_path)
        
        holdout_df_scaled = holdout_df.copy()
        holdout_df_scaled.loc[:, feature_cols] = scaler.transform(holdout_df_scaled[feature_cols])
        
        eval_env = make_vec_env(lambda: TradingEnv(holdout_df_scaled), n_envs=1)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=1, deterministic=True)
        logger.info(f"Split {i} model reward on holdout set: {mean_reward:.2f}")

        if mean_reward > best_reward:
            best_reward = mean_reward
            best_model_path = model_path
            best_scaler_path = scaler_path
            logger.info(f"🏆 New Champion Found! Split {i} is the best performing model for {state_name}.")
        del model, eval_env

    # --- Final Model Deployment ---
    if best_model_path and best_scaler_path:
        shutil.copy(best_model_path, final_model_path)
        shutil.copy(best_scaler_path, final_scaler_path)
        logger.info(f"✅ Champion model for {state_name} deployed to: {final_model_path}")
    else:
        logger.error(f"Could not determine a champion model for {state_name} advisor.")


def run_specialist_training_pipeline(config_path: str):
    setup_logging()
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info(" HIERARCHICAL RL: CE-ALIGNED SPECIALIST FORGING (WALK-FORWARD EDITION) ")
    logger.info("="*80)

    try:
        full_df = pd.read_parquet(paths.PROCESSED_DATA_FILE) 
        ce_state_datasets = create_ce_state_datasets(full_df)

        for state_id, state_df in ce_state_datasets.items():
            state_name = "BULLISH" if state_id == 1 else "BEARISH"
            logger.info("\n" + "="*60)
            logger.info(f" COMMENCING WALK-FORWARD TRAINING FOR: {state_name} ADVISOR ")
            logger.info("="*60)
            
            specialist_save_dir = str(paths.FINAL_MODEL_DIR / f"specialist_{state_name.lower()}")
            os.makedirs(specialist_save_dir, exist_ok=True)
            
            # Create a fresh config instance for each specialist to avoid carrying over modifications
            agent_config = Config(config_path=config_path)

            # =======================================================================
            # ========== بداية التعديل: تطبيق إعدادات خاصة للخبير البيعي ==========
            # =======================================================================
            if state_name == "BEARISH":
                logger.warning("Applying special emergency configuration for BEARISH advisor due to limited data.")
                
                # Manually override the configuration values for this specific run
                agent_config.config['training']['walk_forward']['n_splits'] = 3
                agent_config.config['training']['walk_forward']['min_train_size'] = 2000
                agent_config.config['training']['walk_forward']['test_size'] = 500
                agent_config.config['training']['walk_forward']['timesteps_per_split'] = 30000
                agent_config.config['training']['learning_rate'] = 0.0001 # Slower learning rate for stability
                
                logger.info(f"Overridden Walk-Forward Config for BEARISH: {agent_config.get('training.walk_forward')}")
                logger.info(f"Overridden Learning Rate for BEARISH: {agent_config.get('training.learning_rate')}")
            # =======================================================================
            # ========== نهاية التعديل ==========
            # =======================================================================

            # Run the full walk-forward pipeline for this specialist
            run_specialist_walk_forward(state_df, specialist_save_dir, agent_config, state_name)

        logger.info("\n" + "="*80)
        logger.info("✅ ALL SPECIALIST AGENT FORGING PROTOCOLS COMPLETE.")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"A critical failure occurred during specialist training: {e}", exc_info=True)
        raise

def run_fine_tuning_on_specialists(config: Config):
    """
    Fine-tunes the existing champion models on the most recent data slice,
    using adaptive settings based on data availability for each specialist.
    """
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info(" 🚀 INITIATING ADAPTIVE FINE-TUNING PROTOCOL FOR CHAMPION SPECIALISTS 🚀 ")
    logger.info("="*80)
    
    try:
        full_df = pd.read_parquet(paths.PROCESSED_DATA_FILE)
        ce_state_datasets = create_ce_state_datasets(full_df)
        ft_config = config.get('training.fine_tuning')
        
        for state_id, state_df in ce_state_datasets.items():
            state_name = "BULLISH" if state_id == 1 else "BEARISH"
            logger.info(f"\n--- Fine-tuning {state_name} Specialist ---")
            
            specialist_dir = paths.FINAL_MODEL_DIR / f"specialist_{state_name.lower()}"
            champion_model_path = specialist_dir / "champion_model.zip"
            
            if not champion_model_path.exists():
                logger.warning(f"No champion model found for {state_name}. Skipping fine-tuning.")
                continue

            # =======================================================================
            # ========== بداية التعديل: إعدادات صقل ديناميكية لكل خبير ==========
            # =======================================================================
            if state_name == "BEARISH":
                # Special settings for the Bearish model with limited data
                logger.warning("Applying adaptive fine-tuning settings for BEARISH specialist.")
                
                # Use a smaller percentage of the most recent data
                recent_data_rows = int(len(state_df) * 0.3) # Use last 30% of available data
                
                # Use more conservative learning parameters
                fine_tuning_timesteps = 15000 # Fewer steps to avoid overfitting
                fine_tuning_lr = 5e-6       # Very low learning rate for gentle adjustments

                logger.info(f"BEARISH specialist will use {recent_data_rows} samples, {fine_tuning_timesteps} timesteps, and LR={fine_tuning_lr}.")

            else: # Bullish model with plenty of data
                # Use the standard robust settings from the config file
                recent_data_rows = int(ft_config.get('recent_data_years', 1.5) * 252 * 96)
                fine_tuning_timesteps = ft_config.get('total_timesteps')
                fine_tuning_lr = ft_config.get('learning_rate')
            # =======================================================================
            # ========== نهاية التعديل ==========
            # =======================================================================
            
            if len(state_df) < recent_data_rows:
                logger.warning(f"Not enough recent data for {state_name} ({len(state_df)} available). Using all available data.")
                recent_df = state_df
            else:
                recent_df = state_df.tail(recent_data_rows).copy()

            logger.info(f"Using last {len(recent_df)} data points for fine-tuning.")

            final_scaler = RobustScaler()
            feature_cols = [c for c in recent_df.columns if c not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
            recent_df.loc[:, feature_cols] = final_scaler.fit_transform(recent_df[feature_cols])

            fine_tune_env = make_vec_env(lambda: TradingEnv(recent_df), n_envs=1)
            model = SAC.load(champion_model_path, env=fine_tune_env, device=DEVICE)
            
            model.learning_rate = fine_tuning_lr
            
            logger.info(f"Engaging fine-tuning for {fine_tuning_timesteps} timesteps with learning rate {model.learning_rate}...")
            model.learn(
                total_timesteps=fine_tuning_timesteps,
                reset_num_timesteps=False,
                progress_bar=False
            )
            
            model.save(champion_model_path)
            joblib.dump(final_scaler, specialist_dir / "champion_scaler.joblib")
            
            logger.info(f"✅ Fine-tuning complete for {state_name}. Model and scaler updated.")
            del model, fine_tune_env
            gc.collect()

    except Exception as e:
        logger.error(f"A critical failure occurred during fine-tuning: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_specialist_training_pipeline(config_path='config/config_sac.yaml')