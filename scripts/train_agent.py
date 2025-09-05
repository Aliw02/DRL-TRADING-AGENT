# scripts/train_agent.py
# PROFESSIONAL, WALK-FORWARD TRAINING PIPELINE
import os
import gc
import sys
import joblib
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.preprocessing import RobustScaler
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.logger import setup_logging
from envs.trading_env import TradingEnv
from models.custom_policy import HybridCNNTransformerPolicy
from config.init import Config
from config import paths
from utils.accelerator import DEVICE

logger = setup_logging()

def train_one_segment(train_df: pd.DataFrame, eval_df: pd.DataFrame, save_path_prefix: str, config: Config):
    """
    Trains a single segment of the walk-forward validation process.
    This function is now highly optimized and configured for stability.
    """
    log_dir = os.path.join(save_path_prefix, "logs/")
    model_save_path = os.path.join(save_path_prefix, "models/")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    # Use a single environment to prevent potential deadlocks with complex data
    train_env = make_vec_env(lambda: TradingEnv(train_df), n_envs=1)
    eval_env = make_vec_env(lambda: TradingEnv(eval_df), n_envs=1)

    # EvalCallback is critical for saving the best performing model from the training run
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_save_path),
        log_path=str(log_dir),
        eval_freq=config.get('training.walk_forward.eval_freq', 7500),
        deterministic=True,
        n_eval_episodes=1
    )

    # Define the custom architecture for the agent
    policy_kwargs = dict(
        features_extractor_class=HybridCNNTransformerPolicy,
        features_extractor_kwargs=dict(features_dim=config.get('model.features_dim'))
    )

                             

def run_walk_forward_training(full_df: pd.DataFrame, config: Config):
    """
    Manages the entire walk-forward training process, splitting data,
    scaling features for each split, and calling the training function.
    """
    logger.info("=" * 80)
    logger.info("STARTING WALK-FORWARD TRAINING PROCESS")
    logger.info("=" * 80)

    wf_config = config.get('training.walk_forward')
    n_splits = wf_config['n_splits']
    min_train_size = wf_config['min_train_size']
    test_size = wf_config['test_size']
    holdout_size = test_size

    # Split the data into a main training set and a final holdout set
    training_data_end = len(full_df) - holdout_size
    training_df = full_df.iloc[:training_data_end]
    holdout_df = full_df.iloc[training_data_end:]

    for i in range(n_splits):
        split_num = i + 1
        logger.info(f"\n===== Processing Walk-Forward Split {split_num}/{n_splits} =====")
        split_save_path = os.path.join(paths.WALK_FORWARD_DIR, f"split_{split_num}")

        if os.path.exists(os.path.join(split_save_path, "models/best_model.zip")):
            logger.info(f"Split {split_num} already trained. Skipping.")
            continue

        # Define the training and validation windows for this split
        train_end = min_train_size + i * test_size
        test_end = train_end + test_size

        if test_end > len(training_df):
            logger.warning(f"Not enough data for split {split_num}. Halting process.")
            break

        train_df_split = training_df.iloc[:train_end]
        test_df_split = training_df.iloc[train_end:test_end]

        # --- CRITICAL: Scale data per-split to avoid lookahead bias ---
        scaler = RobustScaler()
        feature_cols = [c for c in training_df.columns if c not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]

        # Fit the scaler ONLY on the training data for this split
        train_df_split[feature_cols] = scaler.fit_transform(train_df_split[feature_cols])
        # Transform the validation data using the same scaler
        test_df_split[feature_cols] = scaler.transform(test_df_split[feature_cols])
        
        # Save the scaler for this specific split
        os.makedirs(split_save_path, exist_ok=True)
        joblib.dump(scaler, os.path.join(split_save_path, "scaler.joblib"))

        train_one_segment(train_df_split, test_df_split, split_save_path, config)
    
    gc.collect()
    logger.info("=" * 80)
    logger.info("Walk-Forward Training Process Completed")
    logger.info("=" * 80)

    return holdout_df


def find_champion_model(holdout_df_cpu: pd.DataFrame, config: Config):
    logger.info("="*80); logger.info("FINDING CHAMPION MODEL"); logger.info("="*80)
    best_model_path, best_reward = None, -float('inf')
    n_splits = config.get('training.walk_forward.n_splits')
    for i in range(1, n_splits + 1):
        model_path = paths.WALK_FORWARD_DIR / f"split_{i}/models/best_model.zip"
        scaler_path = paths.WALK_FORWARD_DIR / f"split_{i}/scaler.joblib"
        if not model_path.exists(): continue
        
        logger.info(f"Evaluating model from split {i}...")
        model = SAC.load(model_path, device=DEVICE)
        scaler = joblib.load(scaler_path)
        feature_cols = [c for c in holdout_df_cpu.columns if c not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
        holdout_scaled_cpu = pd.DataFrame(scaler.transform(holdout_df_cpu[feature_cols]), columns=feature_cols, index=holdout_df_cpu.index)
        holdout_scaled_cpu['close'] = holdout_df_cpu['close']
        holdout_processed = holdout_scaled_cpu.reset_index(drop=True)
        
        eval_env = make_vec_env(lambda: TradingEnv(holdout_processed), n_envs=1)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=1, deterministic=True)
        logger.info(f"Split {i} model reward on holdout set: {mean_reward:.2f}")
        
        if mean_reward > best_reward:
            best_reward, best_model_path = mean_reward, model_path
            logger.info(f"🏆 New champion found! Split {i} is the best so far.")
        
        del model
        del eval_env
        gc.collect()

    if best_model_path is None: raise FileNotFoundError("No trained models found.")
    return best_model_path

def run_finetuning_for_live_trading(best_model_path, full_df_cpu: pd.DataFrame, config: Config):
    logger.info("="*80); logger.info("STARTING FINE-TUNING PROCESS"); logger.info("="*80)
    ft_config = config.get('training.fine_tuning')
    logger.info(f"Loading champion model from: {best_model_path}")
    
    recent_data_rows = ft_config.get('recent_data_years') * 252 * 24 * (60/15)
    recent_df_cpu = full_df_cpu.tail(int(recent_data_rows))
    
    final_scaler = RobustScaler()
    feature_cols = [c for c in recent_df_cpu.columns if c not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
    scaled_features_cpu = final_scaler.fit_transform(recent_df_cpu[feature_cols])

    final_train_df_cpu = pd.DataFrame(scaled_features_cpu, columns=feature_cols, index=recent_df_cpu.index)
    final_train_df_cpu['close'] = recent_df_cpu['close']
    
    final_train_df = final_train_df_cpu.reset_index(drop=True)
    
    final_env = make_vec_env(lambda: TradingEnv(final_train_df), n_envs=1)
    
    model = SAC.load(best_model_path, env=final_env, device=DEVICE)
    model.learning_rate = ft_config.get('learning_rate')
    
    logger.info(f"Starting fine-tuning for {ft_config.get('total_timesteps')} timesteps...")
    model.learn(total_timesteps=ft_config.get('total_timesteps'), reset_num_timesteps=False, progress_bar=True)
    
    model.save(paths.FINAL_MODEL_PATH)
    joblib.dump(final_scaler, paths.FINAL_SCALER_PATH)
    logger.info(f"✅ Fine-tuning complete. Model saved to: {paths.FINAL_MODEL_PATH}"); logger.info("="*80)

def run_agent_training(config: Config):
    data_path = paths.FINAL_ENRICHED_DATA_FILE
    full_df_cpu = pd.read_parquet(data_path)
    holdout_df = run_walk_forward_training(full_df_cpu, config)
    champion_model_path = find_champion_model(holdout_df, config)
    run_finetuning_for_live_trading(champion_model_path, full_df_cpu, config)

def run_agent_finetuning(config: Config):
    data_path = paths.FINAL_ENRICHED_DATA_FILE
    full_df_cpu = pd.read_parquet(data_path)
    holdout_size = config.get('training.walk_forward.test_size')
    holdout_df_cpu = full_df_cpu.iloc[len(full_df_cpu) - holdout_size:]
    champion_model_path = find_champion_model(holdout_df_cpu, config)
    run_finetuning_for_live_trading(champion_model_path, full_df_cpu, config)