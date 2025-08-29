# scripts/train_agent.py

import os
import sys
import pandas as pd
import joblib
import torch
from typing import Callable

# --- Add project root to path ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- Necessary Imports ---
from stable_baselines3 import PPO # Use PPO for custom feature extractor policies
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from sklearn.preprocessing import RobustScaler

from utils.logger import get_logger
from utils.data_transformation import DataTransformer
from envs.trading_env import TradingEnv
from models.custom_policy import CustomActorCriticPolicy # Import our new policy
from config.init import config
from config import paths # Import the centralized paths

logger = get_logger(__name__)

# --- Custom Console Logger Callback (no changes needed) ---
class ConsoleLogCallback(BaseCallback):
    # ... (code remains the same as you provided) ...
    """
    A custom callback that prints evaluation results to the console.
    """
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose)
        self.last_eval_timestep = 0

    def _on_step(self) -> bool:
        if 'eval/mean_reward' in self.logger.name_to_value:
            mean_reward = self.logger.name_to_value['eval/mean_reward']
            latest_eval_step = self.num_timesteps
            
            if latest_eval_step > self.last_eval_timestep:
                self.last_eval_timestep = latest_eval_step
                mean_ep_length = self.logger.name_to_value['eval/mean_ep_length']
                
                if 'eval/mean_trade_count' in self.logger.name_to_value:
                    mean_trade_count = self.logger.name_to_value['eval/mean_trade_count']
                else:
                    mean_trade_count = 0 

                print("\n" + "="*50)
                print(f"CUSTOM EVALUATION REPORT @ Timestep: {self.num_timesteps}")
                print(f"  - Mean Reward:      {mean_reward:.2f}")
                print(f"  - Episode Length:   {mean_ep_length:.0f}")
                print(f"  - Mean Trades:      {mean_trade_count:.2f}")
                print("="*50 + "\n")
        return True


# --- Train One Segment (Updated for new model) ---
def train_one_segment(train_df, eval_df, save_path_prefix):
    """
    Trains a PPO model with the custom CNN-Transformer policy.
    """
    try:
        log_dir = save_path_prefix / "logs/"
        model_save_path = save_path_prefix / "models/"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_save_path, exist_ok=True)

        env = TradingEnv(train_df)
        env = Monitor(env, str(log_dir))
        eval_env = TradingEnv(eval_df)
        eval_env = Monitor(eval_env)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_save_path / "best/"),
            log_path=str(log_dir),
            eval_freq=5000,
            deterministic=True
        )
        console_callback = ConsoleLogCallback()
        
        # --- **CRITICAL CHANGE**: Use the custom policy architecture ---
        policy_kwargs = dict(
            features_extractor_class=CustomActorCriticPolicy,
            features_extractor_kwargs=dict(features_dim=config.get('model.features_dim'))
        )
        
        def linear_schedule(initial_value: float) -> Callable[[float], float]:
            return lambda progress: progress * initial_value
        
        initial_lr = config.get('training.learning_rate', 0.0001)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # --- **CRITICAL CHANGE**: Use PPO, not RecurrentPPO ---
        # The new architecture is a feature extractor, which works best with the standard PPO.
        model = PPO(
            "MlpPolicy", # The policy head on top of our custom features
            env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            learning_rate=linear_schedule(initial_lr),
            n_steps=config.get('training.n_steps', 2048),
            batch_size=config.get('training.batch_size', 64),
            n_epochs=config.get('training.n_epochs', 10),
            gamma=config.get('training.gamma', 0.99),
            tensorboard_log=str(log_dir),
            device=device
        )

        total_timesteps = config.get('training.total_timesteps', 100000)
        logger.info(f"Starting training for {total_timesteps} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, console_callback]
        )
        
        final_model_path = model_save_path / "final_model.zip"
        model.save(str(final_model_path))
        logger.info(f"Training for segment completed. Final model saved to {final_model_path}")
        
        return str(final_model_path)

    except Exception as e:
        logger.error(f"Error during segment training: {e}", exc_info=True)
        raise

# --- Walk-Forward Training (Updated with dynamic paths) ---
def run_walk_forward_training():
    """
    Main function to run the entire walk-forward training process.
    It now loads data from the preprocessed Parquet file.
    """
    print("Starting Walk-Forward Training Process...")
    
    # --- **CRITICAL CHANGE**: Load from the efficient Parquet file ---
    processed_data_path = paths.DATA_DIR / "processed_training_data.parquet"
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(
            f"Preprocessed data file not found at {processed_data_path}. "
            "Please run 'scripts/preprocess_data.py' first!"
        )
    
    print(f"Loading preprocessed data from: {processed_data_path}")
    # Reading from Parquet is much faster and uses less RAM
    full_df = pd.read_parquet(processed_data_path)
    
    wf_config = config.get('training.walk_forward', {})
    n_splits = wf_config.get('n_splits', 5)
    min_train_size = wf_config.get('min_train_size', 60000)
    test_size = wf_config.get('test_size', 15000)
    
    for i in range(n_splits):
        split_num = i + 1
        print(f"===== Processing Walk-Forward Split {split_num}/{n_splits} =====")
        
        split_save_path = paths.WALK_FORWARD_DIR / f"split_{split_num}"
        final_model_path = split_save_path / "models/final_model.zip"

        if os.path.exists(final_model_path):
            print(f"Split {split_num} has already been trained. Skipping.")
            continue

        logger.info(f"----- Starting Training for Split #{split_num} -----")
        
        train_start = 0
        train_end = min_train_size + i * test_size
        test_start = train_end
        test_end = test_start + test_size
        
        if test_end > len(full_df):
            print(f"Not enough data for split {split_num}. Stopping.")
            break
            
        train_df_unscaled = full_df.iloc[train_start:train_end].copy()
        test_df_unscaled = full_df.iloc[test_start:test_end].copy()
        
        logger.info("Scaling features for the current split...")
        scaler = RobustScaler()
        # Separate close price before scaling
        feature_cols = [col for col in train_df_unscaled.columns if 'close' not in col and 'time' not in col]
        
        train_df_scaled = pd.DataFrame(scaler.fit_transform(train_df_unscaled[feature_cols]), columns=feature_cols, index=train_df_unscaled.index)
        train_df_scaled['close'] = train_df_unscaled['close'].values
        
        test_df_scaled = pd.DataFrame(scaler.transform(test_df_unscaled[feature_cols]), columns=feature_cols, index=test_df_unscaled.index)
        test_df_scaled['close'] = test_df_unscaled['close'].values

        train_one_segment(train_df_scaled, test_df_scaled, split_save_path)
        
        scaler_path = split_save_path / "models/robust_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"Scaler for split {split_num} saved to {scaler_path}")

    print("===== Walk-Forward Training Process Completed =====")

# --- Fine-Tuning (Updated with dynamic paths and logic) ---
def run_finetuning_for_live_trading():
    print("===== Starting Fine-Tuning Process for Live Trading =====")
    
    # Find the best model from the last completed split
    last_split_num = config.get('training.walk_forward.n_splits', 5)
    best_model_path = paths.WALK_FORWARD_DIR / f"split_{last_split_num}/models/best/best_model.zip"
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Could not find the best model from the last split at: {best_model_path}")
    print(f"Identified best model to fine-tune: {best_model_path}")

    transformer = DataTransformer()
    full_df = transformer.load_and_preprocess_data(file_path=str(paths.TRAIN_DATA_FILE))

    # --- **CRITICAL CHANGE**: Use only the last 2-3 years for fine-tuning ---
    # Assuming M1 data has ~525,600 rows per year. Let's take ~2.5 years.
    recent_data_rows = 525600 * 3.0
    recent_df = full_df.tail(int(recent_data_rows)).copy()

    print(f"Using the last {len(recent_df)} records (approx. 3 years) for fine-tuning.")

    # Scale only the recent data
    scaler = RobustScaler()
    feature_cols = [col for col in recent_df.columns if 'close' not in col and 'time' not in col]
    
    scaled_features = scaler.fit_transform(recent_df[feature_cols])
    final_train_df = pd.DataFrame(scaled_features, columns=feature_cols, index=recent_df.index)
    final_train_df['close'] = recent_df['close'].values
    
    print(f"Prepared recent dataset with {len(final_train_df)} samples for fine-tuning.")


    final_env = TradingEnv(final_train_df)
    final_env = Monitor(final_env)

    finetune_lr = 1e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading pre-trained model on device '{device}' and starting fine-tuning...")
    model = PPO.load(best_model_path, env=final_env, device=device)
    model.learning_rate = finetune_lr
    
    model.learn(total_timesteps=200000, reset_num_timesteps=False)

    # Save the final model and the scaler used for the *entire* dataset
    model.save(str(paths.FINAL_MODEL_PATH))
    joblib.dump(scaler, str(paths.FINAL_SCALER_PATH))
    
    print("===== Fine-Tuning Completed =====")
    print(f"Your fine-tuned model for live trading is ready at: {paths.FINAL_MODEL_PATH}")
    print(f"The final data scaler is ready at: {paths.FINAL_SCALER_PATH}")

# --- Main execution block ---
if __name__ == "__main__":
    run_walk_forward_training()
    run_finetuning_for_live_trading()