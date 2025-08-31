# scripts/train_agent.py (COMPLETELY RE-ARCHITECTED FOR WALK-FORWARD & FINE-TUNING)
import os, sys, pandas as pd, joblib, torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from sklearn.preprocessing import RobustScaler

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logger import setup_logging, get_logger
from envs.trading_env import TradingEnv
from models.custom_policy import CustomActorCriticPolicy
from config.init import Config
from config import paths

logger = setup_logging()

def train_one_segment(train_df: pd.DataFrame, eval_df: pd.DataFrame, save_path_prefix: str, config: Config):
    """Trains a single SAC model segment for the walk-forward process."""
    log_dir = save_path_prefix / "logs/"
    model_save_path = save_path_prefix / "models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    train_env = Monitor(TradingEnv(train_df), str(log_dir))
    eval_env = Monitor(TradingEnv(eval_df), str(log_dir) + "_eval")

    eval_callback = EvalCallback(eval_env, best_model_save_path=str(model_save_path),
                                 log_path=str(log_dir), eval_freq=10000, deterministic=True)

    policy_kwargs = dict(features_extractor_class=CustomActorCriticPolicy,
                       features_extractor_kwargs=dict(features_dim=config.get('model.features_dim')))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SAC("MlpPolicy", train_env, policy_kwargs=policy_kwargs, verbose=0,
                learning_rate=config.get('training.learning_rate'),
                buffer_size=config.get('training.buffer_size'),
                batch_size=config.get('training.batch_size'),
                tau=config.get('training.tau'), gamma=config.get('training.gamma'),
                train_freq=tuple(config.get('training.train_freq')),
                learning_starts=config.get('training.learning_starts'),
                tensorboard_log=str(log_dir), device=device)

    total_timesteps = config.get('training.walk_forward.timesteps_per_split')
    logger.info(f"Training segment for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    logger.info(f"Segment training complete. Best model saved in {model_save_path}")

def run_walk_forward_training(full_df: pd.DataFrame, config: Config):
    """Manages the entire walk-forward training process."""
    logger.info("="*80); logger.info("STARTING WALK-FORWARD TRAINING PROCESS"); logger.info("="*80)
    
    wf_config = config.get('training.walk_forward')
    n_splits = wf_config.get('n_splits')
    min_train_size = wf_config.get('min_train_size')
    test_size = wf_config.get('test_size')
    
    for i in range(n_splits):
        split_num = i + 1
        logger.info(f"\n===== Processing Walk-Forward Split {split_num}/{n_splits} =====")
        
        split_save_path = paths.WALK_FORWARD_DIR / f"split_{split_num}"
        if (split_save_path / "models/best_model.zip").exists():
            logger.info(f"Split {split_num} already trained. Skipping.")
            continue

        train_end = min_train_size + i * test_size
        test_end = train_end + test_size
        if test_end > len(full_df):
            logger.warning(f"Not enough data for split {split_num}. Stopping walk-forward.")
            break
            
        train_df_unscaled = full_df.iloc[:train_end].copy()
        test_df_unscaled = full_df.iloc[train_end:test_end].copy()
        
        scaler = RobustScaler()
        feature_cols = [c for c in full_df.columns if c not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
        
        train_df_scaled = pd.DataFrame(scaler.fit_transform(train_df_unscaled[feature_cols]), columns=feature_cols, index=train_df_unscaled.index)
        train_df_scaled['close'] = train_df_unscaled['close']
        
        test_df_scaled = pd.DataFrame(scaler.transform(test_df_unscaled[feature_cols]), columns=feature_cols, index=test_df_unscaled.index)
        test_df_scaled['close'] = test_df_unscaled['close']

        train_one_segment(train_df_scaled, test_df_scaled, split_save_path, config)
        joblib.dump(scaler, split_save_path / "scaler.joblib")
    
    logger.info("="*80); logger.info("Walk-Forward Training Process Completed"); logger.info("="*80)

def run_finetuning_for_live_trading(full_df, config: Config):
    """Finds the best model from the last split and fine-tunes it on recent data."""
    logger.info("="*80); logger.info("STARTING FINE-TUNING PROCESS FOR PRODUCTION MODEL"); logger.info("="*80)
    
    ft_config = config.get('training.fine_tuning')
    n_splits = config.get('training.walk_forward.n_splits')
    
    # Find the last successfully trained split
    last_trained_split = 0
    for i in range(n_splits, 0, -1):
        if (paths.WALK_FORWARD_DIR / f"split_{i}/models/best_model.zip").exists():
            last_trained_split = i
            break
            
    if last_trained_split == 0:
        raise FileNotFoundError("No trained models found from walk-forward splits. Cannot fine-tune.")

    logger.info(f"Found best model from last completed split: split_{last_trained_split}")
    best_model_path = paths.WALK_FORWARD_DIR / f"split_{last_trained_split}/models/best_model.zip"
    
    # Prepare recent data for fine-tuning
    recent_data_rows = ft_config.get('recent_data_years') * 252 * 24 * 60 # Approx rows in N years of M1 data
    recent_df = full_df.tail(int(recent_data_rows)).copy()
    logger.info(f"Using last {len(recent_df)} records (approx. {ft_config.get('recent_data_years')} years) for fine-tuning.")
    
    # Scale recent data with a new, final scaler
    final_scaler = RobustScaler()
    feature_cols = [c for c in recent_df.columns if c not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
    scaled_features = final_scaler.fit_transform(recent_df[feature_cols])
    final_train_df = pd.DataFrame(scaled_features, columns=feature_cols, index=recent_df.index)
    final_train_df['close'] = recent_df['close']
    
    # Load the best model and fine-tune it
    final_env = Monitor(TradingEnv(final_train_df))
    model = SAC.load(best_model_path, env=final_env, device="cuda" if torch.cuda.is_available() else "cpu")
    model.learning_rate = ft_config.get('learning_rate')
    
    logger.info(f"Starting fine-tuning for {ft_config.get('total_timesteps')} timesteps with learning rate {model.learning_rate}...")
    model.learn(total_timesteps=ft_config.get('total_timesteps'), reset_num_timesteps=False)
    
    # Save the final production-ready artifacts
    model.save(paths.FINAL_MODEL_PATH)
    joblib.dump(final_scaler, paths.FINAL_SCALER_PATH)
    
    logger.info(f"✅ Fine-tuning complete. Production model saved to: {paths.FINAL_MODEL_PATH}")
    logger.info(f"Production scaler saved to: {paths.FINAL_SCALER_PATH}"); logger.info("="*80)

def run_agent_training(config: Config):
    """The main entry point that runs the full training and fine-tuning pipeline."""
    data_path = paths.FINAL_ENRICHED_DATA_FILE
    if not data_path.exists():
        raise FileNotFoundError("Enriched data file not found. Run the main pipeline.")
    
    full_df = pd.read_parquet(data_path)
    
    # Run the two main stages
    run_walk_forward_training(full_df, config)
    run_finetuning_for_live_trading(full_df, config)