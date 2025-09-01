# scripts/train_agent.py (FINAL DYNAMIC & OPTIMIZED VERSION)
import os
import sys
import joblib
import torch
import numpy as np
import pandas as pd_cpu  # Use an alias for the CPU version of pandas

from utils.accelerator import pd, to_gpu, DEVICE
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.preprocessing import RobustScaler

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logger import setup_logging
from envs.trading_env import TradingEnv
from models.custom_policy import CustomActorCriticPolicy
from config.init import Config
from config import paths

logger = setup_logging()

# (The train_one_segment function remains the same as the previous version)
def train_one_segment(train_df, eval_df, save_path_prefix: str, config: Config):
    log_dir = save_path_prefix / "logs/"
    model_save_path = save_path_prefix / "models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    train_env = make_vec_env(lambda: TradingEnv(train_df), n_envs=8)
    eval_env = make_vec_env(lambda: TradingEnv(eval_df), n_envs=1)

    eval_callback = EvalCallback(eval_env, best_model_save_path=str(model_save_path),
                                 log_path=str(log_dir), eval_freq=10000, deterministic=True)

    policy_kwargs = dict(features_extractor_class=CustomActorCriticPolicy,
                       features_extractor_kwargs=dict(features_dim=config.get('model.features_dim')))
    
    model = SAC("MlpPolicy", train_env, policy_kwargs=policy_kwargs, verbose=0,
                learning_rate=config.get('training.learning_rate'),
                buffer_size=config.get('training.buffer_size'),
                batch_size=config.get('training.batch_size'),
                tau=config.get('training.tau'), gamma=config.get('training.gamma'),
                train_freq=tuple(config.get('training.train_freq')),
                learning_starts=config.get('training.learning_starts'),
                tensorboard_log=str(log_dir), device="cuda", use_sde=True)

    total_timesteps = config.get('training.walk_forward.timesteps_per_split')
    logger.info(f"🚀 Starting training for {total_timesteps} timesteps on {train_env.num_envs} parallel environments...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    logger.info(f"✅ Segment training complete. Best model saved in {model_save_path}")

def run_walk_forward_training(full_df_cpu, config: Config):
    logger.info("="*80); logger.info("STARTING WALK-FORWARD TRAINING PROCESS"); logger.info("="*80)
    
    wf_config = config.get('training.walk_forward')
    n_splits, min_train_size, test_size = wf_config['n_splits'], wf_config['min_train_size'], wf_config['test_size']
    
    # NEW: Define a final hold-out set for champion selection
    holdout_size = test_size
    training_data_end = len(full_df_cpu) - holdout_size
    training_df_cpu = full_df_cpu.iloc[:training_data_end]
    holdout_df_cpu = full_df_cpu.iloc[training_data_end:]

    for i in range(n_splits):
        split_num = i + 1
        logger.info(f"\n===== Processing Walk-Forward Split {split_num}/{n_splits} =====")
        split_save_path = paths.WALK_FORWARD_DIR / f"split_{split_num}"
        if (split_save_path / "models/best_model.zip").exists():
            logger.info(f"Split {split_num} already trained. Skipping.")
            continue

        train_end = min_train_size + i * test_size
        test_end = train_end + test_size

        if test_end > len(training_df_cpu):
            logger.warning(f"Not enough data for split {split_num}. Stopping walk-forward.")
            break
        
        train_df_cpu_split, test_df_cpu_split = training_df_cpu.iloc[:train_end], training_df_cpu.iloc[train_end:test_end]
        
        scaler = RobustScaler()
        feature_cols = [c for c in training_df_cpu.columns if c not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
        
        train_df_scaled_cpu = pd_cpu.DataFrame(scaler.fit_transform(train_df_cpu_split[feature_cols]), columns=feature_cols, index=train_df_cpu_split.index)
        train_df_scaled_cpu['close'] = train_df_cpu_split['close']
        
        test_df_scaled_cpu = pd_cpu.DataFrame(scaler.transform(test_df_cpu_split[feature_cols]), columns=feature_cols, index=test_df_cpu_split.index)
        test_df_scaled_cpu['close'] = test_df_cpu_split['close']
        
        # FIX: Reset the index before converting to GPU DataFrame to ensure a simple integer index
        train_df_processed, test_df_processed = to_gpu(train_df_scaled_cpu.reset_index(drop=True)), to_gpu(test_df_scaled_cpu.reset_index(drop=True))

        train_one_segment(train_df_processed, test_df_processed, split_save_path, config)
        joblib.dump(scaler, split_save_path / "scaler.joblib")
    
    logger.info("="*80); logger.info("Walk-Forward Training Process Completed"); logger.info("="*80)
    return holdout_df_cpu # Return the holdout set for the next step

def find_champion_model(holdout_df_cpu, config: Config):
    """Evaluates all split models on a holdout set to find the best one."""
    logger.info("="*80); logger.info("FINDING CHAMPION MODEL FROM ALL SPLITS"); logger.info("="*80)
    best_model_path = None
    best_reward = -np.inf
    n_splits = config.get('training.walk_forward.n_splits')

    for i in range(1, n_splits + 1):
        model_path = paths.WALK_FORWARD_DIR / f"split_{i}/models/best_model.zip"
        scaler_path = paths.WALK_FORWARD_DIR / f"split_{i}/scaler.joblib"
        if not model_path.exists():
            continue

        logger.info(f"Evaluating model from split {i}...")
        model = SAC.load(model_path)
        scaler = joblib.load(scaler_path)

        feature_cols = [c for c in holdout_df_cpu.columns if c not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
        holdout_scaled_cpu = pd_cpu.DataFrame(scaler.transform(holdout_df_cpu[feature_cols]), columns=feature_cols, index=holdout_df_cpu.index)
        holdout_scaled_cpu['close'] = holdout_df_cpu['close']
        
        # FIX: Reset the index here as well
        holdout_processed = to_gpu(holdout_scaled_cpu.reset_index(drop=True))

        eval_env = make_vec_env(lambda: TradingEnv(holdout_processed), n_envs=1)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=1, deterministic=True)
        logger.info(f"Split 1 model reward on holdout set: {mean_reward}")

        if mean_reward > best_reward:
            best_reward = mean_reward
            best_model_path = model_path
            logger.info(f"🏆 New champion found! Split {i} is the best so far.")

    if best_model_path is None:
        raise FileNotFoundError("Could not find any trained models to evaluate.")
    
    return best_model_path

def run_finetuning_for_live_trading(best_model_path, full_df_cpu, config: Config):
    """Finds the best model and fine-tunes it on recent data."""
    logger.info("="*80); logger.info("STARTING FINE-TUNING PROCESS FOR PRODUCTION MODEL"); logger.info("="*80)
    
    ft_config = config.get('training.fine_tuning')
    
    logger.info(f"Loading champion model from: {best_model_path}")
    
    recent_data_rows = ft_config.get('recent_data_years') * 252 * 24 * 60
    recent_df_cpu = full_df_cpu.tail(int(recent_data_rows))
    
    final_scaler = RobustScaler()
    feature_cols = [c for c in recent_df_cpu.columns if c not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
    scaled_features_cpu = final_scaler.fit_transform(recent_df_cpu[feature_cols])
    
    # We must use the CPU version of pandas here as it was imported with an alias
    final_train_df_cpu = pd_cpu.DataFrame(scaled_features_cpu, columns=feature_cols, index=recent_df_cpu.index)
    final_train_df_cpu['close'] = recent_df_cpu['close']
    
    # FIX: Reset the index before converting to GPU DataFrame
    final_train_df = to_gpu(final_train_df_cpu.reset_index(drop=True))
    
    final_env = make_vec_env(lambda: TradingEnv(final_train_df), n_envs=8)
    
    # --- DYNAMIC DEVICE SELECTION ---
    # The device is now set dynamically based on GPU availability.
    model = SAC.load(best_model_path, env=final_env, device=DEVICE)
    
    # The learning rate is already set to a low value from the config file.
    model.learning_rate = ft_config.get('learning_rate')
    
    logger.info(f"Starting fine-tuning for {ft_config.get('total_timesteps')} timesteps with learning rate {model.learning_rate} on device '{DEVICE}'...")
    model.learn(total_timesteps=ft_config.get('total_timesteps'), reset_num_timesteps=False)
    
    model.save(paths.FINAL_MODEL_PATH)
    joblib.dump(final_scaler, paths.FINAL_SCALER_PATH)
    
    logger.info(f"✅ Fine-tuning complete. Production model saved to: {paths.FINAL_MODEL_PATH}")
    logger.info(f"Production scaler saved to: {paths.FINAL_SCALER_PATH}"); logger.info("="*80)


def run_agent_training(config: Config):
    data_path = paths.FINAL_ENRICHED_DATA_FILE
    full_df_cpu = pd_cpu.read_parquet(data_path)
    
    # Stage 1: Run walk-forward training and get the holdout set
    holdout_df = run_walk_forward_training(full_df_cpu, config)
    
    # Stage 2: Find the champion model from all splits
    champion_model_path = find_champion_model(holdout_df, config)

    # Stage 3: Fine-tune the champion model for production
    run_finetuning_for_live_trading(champion_model_path, full_df_cpu, config)



def run_agent_finetuning(config: Config):
    # This is the new logic to find the best model from the pre-trained splits
    # and then fine-tune it, skipping the main training loop.
    data_path = paths.FINAL_ENRICHED_DATA_FILE
    full_df_cpu = pd_cpu.read_parquet(data_path)
    
    # Define a final hold-out set for champion selection
    holdout_size = config.get('training.walk_forward.test_size')
    holdout_df_cpu = full_df_cpu.iloc[len(full_df_cpu) - holdout_size:]

    # Stage 1: Find the champion model from all splits (pre-trained)
    champion_model_path = find_champion_model(holdout_df_cpu, config)

    # Stage 2: Fine-tune the champion model for production
    run_finetuning_for_live_trading(champion_model_path, full_df_cpu, config)
