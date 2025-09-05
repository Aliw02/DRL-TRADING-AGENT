# scripts/train_agent.py
# FINAL, BATTLE-TESTED, AND COMPLETE TRAINING PROTOCOL

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

# --- Add project root to path for robust imports ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logger import setup_logging, get_logger
from envs.trading_env import TradingEnv
from models.custom_policy import HybridCNNTransformerPolicy
from config.init import Config
from config import paths
from utils.accelerator import DEVICE

def train_one_segment(train_df: pd.DataFrame, eval_df: pd.DataFrame, save_path_prefix: str, config: Config):
    """
    Trains a single segment (or a single specialist agent). This is the core
    engine of our entire training operation. This corrected version includes
    the model instantiation and the critical model.learn() call.
    """
    logger = get_logger(__name__)
    logger.info(f"========= COMMENCING TRAINING SEGMENT =========")
    
    log_dir = os.path.join(save_path_prefix, "logs/")
    model_save_path = os.path.join(save_path_prefix, "models/")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    # Use a single environment to prevent potential deadlocks with complex data
    train_env = make_vec_env(lambda: TradingEnv(train_df), n_envs=1)
    eval_env = make_vec_env(lambda: TradingEnv(eval_df), n_envs=1)

    # EvalCallback is our mechanism for saving only the best-performing model
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

    # --- CRITICAL FIX: The model instantiation and learning call were missing ---
    # Instantiate the SAC agent with our professional-grade parameters
    model = SAC(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=config.get('training.learning_rate'),
        buffer_size=config.get('training.buffer_size'),
        batch_size=config.get('training.batch_size'),
        tau=config.get('training.tau'),
        gamma=config.get('training.gamma'),
        gradient_steps=config.get('training.gradient_steps'),
        train_freq=tuple(config.get('training.train_freq')),
        learning_starts=config.get('training.learning_starts'),
        tensorboard_log=str(log_dir),
        device=DEVICE,
        use_sde=True
    )

    total_timesteps = config.get('training.walk_forward.timesteps_per_split')
    logger.info(f"🚀 Engaging training protocol for {total_timesteps} timesteps...")
    
    # --- The IGNITION SWITCH ---
    # This is the call that actually starts the learning process.
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True # Provide visual feedback on training progress
    )
    
    logger.info(f"✅ Segment training complete.")
    
    # Explicit memory management
    del model, train_env, eval_env
    gc.collect()

def run_walk_forward_training(full_df: pd.DataFrame, config: Config):
    """
    Manages the entire walk-forward training process, splitting data,
    scaling features for each split, and calling the training function.
    """
    logger = get_logger(__name__)
    logger.info("=" * 80)
    logger.info("STARTING WALK-FORWARD TRAINING PROCESS")
    logger.info("=" * 80)

    wf_config = config.get('training.walk_forward')
    n_splits = wf_config['n_splits']
    min_train_size = wf_config['min_train_size']
    test_size = wf_config['test_size']
    holdout_size = test_size

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

        train_end = min_train_size + i * test_size
        test_end = train_end + test_size

        if test_end > len(training_df):
            logger.warning(f"Not enough data for split {split_num}. Halting process.")
            break

        train_df_split = training_df.iloc[:train_end]
        test_df_split = training_df.iloc[train_end:test_end]

        scaler = RobustScaler()
        feature_cols = [c for c in training_df.columns if c not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]

        train_df_split.loc[:, feature_cols] = scaler.fit_transform(train_df_split[feature_cols])
        test_df_split.loc[:, feature_cols] = scaler.transform(test_df_split[feature_cols])
        
        os.makedirs(split_save_path, exist_ok=True)
        joblib.dump(scaler, os.path.join(split_save_path, "scaler.joblib"))

        train_one_segment(train_df_split, test_df_split, split_save_path, config)
    
    gc.collect()
    logger.info("=" * 80)
    logger.info("Walk-Forward Training Process Completed")
    logger.info("=" * 80)

    return holdout_df


def find_champion_model(holdout_df: pd.DataFrame, config: Config):
    logger = get_logger(__name__)
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
        
        feature_cols = [c for c in holdout_df.columns if c not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
        holdout_df.loc[:, feature_cols] = scaler.transform(holdout_df[feature_cols])
        
        eval_env = make_vec_env(lambda: TradingEnv(holdout_df), n_envs=1)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=1, deterministic=True)
        logger.info(f"Split {i} model reward on holdout set: {mean_reward:.2f}")
        
        if mean_reward > best_reward:
            best_reward, best_model_path = mean_reward, model_path
            logger.info(f"🏆 New champion found! Split {i} is the best so far.")
        
        del model, eval_env
        gc.collect()

    if best_model_path is None: raise FileNotFoundError("No trained models found after walk-forward validation.")
    return best_model_path

def run_finetuning_for_live_trading(best_model_path, full_df: pd.DataFrame, config: Config):
    logger = get_logger(__name__)
    logger.info("="*80); logger.info("STARTING FINE-TUNING PROCESS FOR PRODUCTION DEPLOYMENT"); logger.info("="*80)
    ft_config = config.get('training.fine_tuning')
    logger.info(f"Loading champion model from: {best_model_path}")
    
    recent_data_rows = int(ft_config.get('recent_data_years') * 252 * 24 * 4) # Assuming 15min data
    recent_df = full_df.tail(recent_data_rows)
    
    final_scaler = RobustScaler()
    feature_cols = [c for c in recent_df.columns if c not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
    recent_df.loc[:, feature_cols] = final_scaler.fit_transform(recent_df[feature_cols])
    
    final_env = make_vec_env(lambda: TradingEnv(recent_df), n_envs=1)
    
    model = SAC.load(best_model_path, env=final_env, device=DEVICE)
    model.learning_rate = ft_config.get('learning_rate')
    
    logger.info(f"Starting fine-tuning for {ft_config.get('total_timesteps')} timesteps...")
    model.learn(total_timesteps=ft_config.get('total_timesteps'), reset_num_timesteps=False, progress_bar=True)
    
    model.save(paths.FINAL_MODEL_PATH)
    joblib.dump(final_scaler, paths.FINAL_SCALER_PATH)
    logger.info(f"✅ Fine-tuning complete. Production model saved to: {paths.FINAL_MODEL_PATH}"); logger.info("="*80)




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