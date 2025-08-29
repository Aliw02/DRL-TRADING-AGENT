import os
import sys
import pandas as pd
import joblib
import torch
from typing import Callable

# --- استيراد المكتبات الضرورية ---
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from sklearn.preprocessing import RobustScaler

# --- إضافة مسار المشروع للوصول إلى الوحدات الأخرى ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logger import get_logger
from utils.data_transformation import DataTransformer
from envs.trading_env import TradingEnv
from config.init import config

logger = get_logger(__name__)

# --- المراقب المخصص لطباعة النتائج على الشاشة ---
class ConsoleLogCallback(BaseCallback):
    """
    A custom callback that prints evaluation results to the console
    by reading them directly from the logger after an evaluation run.
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

# --- دالة تدريب جزء واحد ---
def train_one_segment(train_df, eval_df, save_path_prefix):
    """
    Trains a RecurrentPPO (LSTM) model on a single segment of data.
    """
    try:
        log_dir = os.path.join(save_path_prefix, "logs/")
        model_save_path = os.path.join(save_path_prefix, "models/")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_save_path, exist_ok=True)

        env = TradingEnv(train_df)
        env = Monitor(env, log_dir)
        eval_env = TradingEnv(eval_df)
        eval_env = Monitor(eval_env)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_save_path, "best/"),
            log_path=log_dir,
            eval_freq=5000,
            deterministic=True
        )
        console_callback = ConsoleLogCallback()
        
        policy_kwargs = dict(
            n_lstm_layers=2,
            lstm_hidden_size=512,
        )
        
        def linear_schedule(initial_value: float) -> Callable[[float], float]:
            def func(progress_remaining: float) -> float:
                return progress_remaining * initial_value
            return func
        
        initial_lr = config.get('training.learning_rate', 0.0001)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            learning_rate=linear_schedule(initial_lr),
            n_steps=config.get('training.n_steps', 4096),
            batch_size=config.get('training.batch_size', 64),
            n_epochs=config.get('training.n_epochs', 10),
            gamma=config.get('training.gamma', 0.99),
            ent_coef=config.get('training.ent_coef', 0.05),
            tensorboard_log=log_dir,
            device=device
        )

        total_timesteps = config.get('training.total_timesteps', 1200000)
        logger.info(f"Starting training for {total_timesteps} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, console_callback]
        )
        
        final_model_path = os.path.join(model_save_path, "final_recurrent_model.zip")
        model.save(final_model_path)
        logger.info(f"Training for segment completed. Final model saved to {final_model_path}")
        
        return final_model_path

    except Exception as e:
        print(f"Error during segment training: {e}")
        raise

# --- دالة التدريب المتقدم المتتالي (Walk-Forward) ---
def run_walk_forward_training():
    """
    Main function to run the entire walk-forward training process.
    This version is resumable and checks for completed splits.
    """
    print("Starting Walk-Forward Training Process...")
    
    # file_path = config.get('data.file_path')
    timeframe = 1
    file_path = f"/content/drive/MyDrive/DRL_Model_DeepSeek/data/XAUUSDM{timeframe}-FULL.csv"
    transformer = DataTransformer()
    full_df = transformer.load_and_preprocess_data(file_path=file_path)
    
    wf_config = config.get('training.walk_forward', {})
    n_splits = wf_config.get('n_splits', 5)
    min_train_size = wf_config.get('min_train_size', 60000)
    test_size = wf_config.get('test_size', 15000)
    
    for i in range(n_splits):
        split_num = i + 1
        print(f"===== Checking Status for Walk-Forward Split {split_num}/{n_splits} =====")
        
        split_save_path = f"results/walk_forward/split_{split_num}/"
        final_model_path = os.path.join(split_save_path, "models/final_recurrent_model.zip")

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
        
        logger.info("Scaling features for the current split using RobustScaler...")
        scaler = RobustScaler()
        feature_cols = [col for col in train_df_unscaled.columns if col != 'close']
        scaler.fit(train_df_unscaled[feature_cols])
        
        train_df_scaled = pd.DataFrame(scaler.transform(train_df_unscaled[feature_cols]), columns=feature_cols)
        train_df_scaled['close'] = train_df_unscaled['close'].values
        test_df_scaled = pd.DataFrame(scaler.transform(test_df_unscaled[feature_cols]), columns=feature_cols)
        test_df_scaled['close'] = test_df_unscaled['close'].values

        train_one_segment(train_df_scaled, test_df_scaled, split_save_path)
        
        model_save_path = os.path.join(split_save_path, "models/")
        scaler_path = os.path.join(model_save_path, "robust_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler for split {split_num} saved to {scaler_path}")

    print("===== Walk-Forward Training Process Completed =====")

# --- دالة الضبط الدقيق للنموذج النهائي ---
def run_finetuning_for_live_trading():
    """
    Finds the best model from the walk-forward splits, loads it,
    and continues training (fine-tunes) it on the entire dataset.
    """
    print("===== Starting Fine-Tuning Process for Live Trading =====")
    
    best_model_path = "results/walk_forward/split_5/models/best/best_model.zip"
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Could not find the best model from the last split at: {best_model_path}")
    print(f"Identified best model to fine-tune: {best_model_path}")

    # file_path = config.get('data.file_path')
    # file_path = config.get('data.file_path')
    timeframe = 1
    file_path = f"/content/drive/MyDrive/DRL_Model_DeepSeek/data/XAUUSDM{timeframe}-FULL.csv"
    
    transformer = DataTransformer()
    full_df = transformer.load_and_preprocess_data(file_path=file_path)
    
    scaler = RobustScaler()
    feature_cols = [col for col in full_df.columns if col != 'close']
    scaled_features = scaler.fit_transform(full_df[feature_cols])
    final_train_df = pd.DataFrame(scaled_features, columns=feature_cols)
    final_train_df['close'] = full_df['close'].values
    print(f"Prepared full dataset with {len(final_train_df)} samples for fine-tuning.")

    final_env = TradingEnv(final_train_df)
    final_env = Monitor(final_env)

    finetune_lr = 1e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading pre-trained model on device '{device}' and starting fine-tuning...")
    model = RecurrentPPO.load(best_model_path, env=final_env, device=device)
    model.learning_rate = finetune_lr
    
    model.learn(total_timesteps=200000)

    final_model_save_path = "results/final_model_for_live/finetuned_model.zip"
    os.makedirs(os.path.dirname(final_model_save_path), exist_ok=True)
    model.save(final_model_save_path)
    
    scaler_path = "results/final_model_for_live/final_robust_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    
    print("===== Fine-Tuning Completed =====")
    print(f"Your fine-tuned model for live trading is ready at: {final_model_save_path}")

# --- نقطة انطلاق البرنامج ---
if __name__ == "__main__":
    run_walk_forward_training()
    run_finetuning_for_live_trading()