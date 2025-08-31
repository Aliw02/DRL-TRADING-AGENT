# scripts/analyze_features.py (COMPLETELY REVISED AND UPGRADED)

import pandas as pd
import joblib
import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# --- Add project root to path ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import SAC
from sklearn.inspection import permutation_importance
from utils.data_transformation import DataTransformer
from utils.logger import setup_logging, get_logger
from config import paths

def analyze_model_features():
    """
    Loads the final trained agent and runs an advanced feature analysis
    using Permutation Importance to identify which features most influence
    the agent's decisions.
    """
    setup_logging()
    logger = get_logger(__name__)

    logger.info("="*80)
    logger.info("STARTING ADVANCED FEATURE ANALYSIS PROCESS")
    logger.info("="*80)
    
    try:
        # --- 1. Load all production artifacts ---
        logger.info("Loading production agent, scalers, and regime model...")
        agent_model = SAC.load(str(paths.FINAL_MODEL_PATH), device='cpu') # Load to CPU for analysis
        agent_scaler = joblib.load(str(paths.FINAL_SCALER_PATH))
        regime_model = joblib.load(paths.FINAL_MODEL_DIR / "regime_gmm_model.joblib")
        regime_scaler = joblib.load(paths.FINAL_MODEL_DIR / "regime_robust_scaler.joblib")

        # --- 2. Prepare unseen data using the full enrichment pipeline ---
        logger.info(f"Loading and processing unseen data from: {paths.BACKTEST_DATA_FILE}")
        transformer = DataTransformer()
        backtest_df = transformer.load_and_preprocess_data(file_path=str(paths.BACKTEST_DATA_FILE))

        regime_features = ['adx', 'bb_width', 'roc_norm', 'rsi_x_adx', 'atr']
        X_raw_regime = backtest_df[regime_features].ffill().bfill()
        X_scaled_regime = regime_scaler.transform(X_raw_regime)
        regime_probabilities = regime_model.predict_proba(X_scaled_regime)
        prob_cols = [f'regime_prob_{i}' for i in range(regime_model.n_components)]
        prob_df = pd.DataFrame(regime_probabilities, index=backtest_df.index, columns=prob_cols)
        enriched_df = backtest_df.join(prob_df)

        # --- 3. Scale the final feature set for the agent ---
        feature_cols = [col for col in enriched_df.columns if col not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
        X_scaled_agent = agent_scaler.transform(enriched_df[feature_cols])
        
        # We need to create sequences for the model's observation space
        num_samples = 1000 # Use a subset for faster analysis
        sequence_length = agent_model.policy.observation_space.shape[0]
        
        # Ensure we have enough data
        if len(X_scaled_agent) < sequence_length + num_samples:
            raise ValueError("Not enough data in the backtest file to create samples for analysis.")

        # Create sequences of shape (n_samples, sequence_length, n_features)
        X_sequences = np.array([X_scaled_agent[i:i+sequence_length] for i in range(num_samples)])
        
        # Add a placeholder for the position feature (last column)
        position_placeholder = np.zeros((X_sequences.shape[0], sequence_length, 1))
        X_sequences_with_pos = np.concatenate([X_sequences, position_placeholder], axis=2).astype(np.float32)

        logger.info(f"Prepared {num_samples} observation sequences for analysis.")

        # --- 4. Permutation Feature Importance ---
        logger.info("Calculating Permutation Feature Importance (this may take a moment)...")
        
        # The scoring function should reflect the agent's goal.
        # We'll use the model's action probabilities (entropy) or value function.
        # Here, we define a simple scorer based on the model's raw action prediction.
        def scoring_fn(estimator, X):
            obs_tensor = torch.as_tensor(X, device='cpu')
            # Get the actions the model would take
            actions, _ = estimator.predict(obs_tensor, deterministic=True)
            return actions

        # We need a custom permutation function to handle the sequence data structure
        def custom_permutation(X, col_to_permute):
            X_permuted = X.copy()
            # Permute the values of the feature across all samples and all timesteps in the sequence
            X_permuted[:, :, col_to_permute] = np.random.permutation(X_permuted[:, :, col_to_permute])
            return X_permuted

        # Calculate importance for each feature
        baseline_predictions = scoring_fn(agent_model.policy, X_sequences_with_pos)
        importances = []
        for i in range(len(feature_cols)):
            X_permuted = custom_permutation(X_sequences_with_pos, i)
            permuted_predictions = scoring_fn(agent_model.policy, X_permuted)
            # Calculate the drop in performance (e.g., mean squared error of actions)
            score_drop = np.mean((baseline_predictions - permuted_predictions)**2)
            importances.append(score_drop)
            
        importances = np.array(importances)
        sorted_idx = importances.argsort()

        # --- 5. Display and Save Results ---
        perm_df = pd.DataFrame(
            data=importances[sorted_idx],
            index=np.array(feature_cols)[sorted_idx],
            columns=['Importance']
        )
        
        print("\n--- Feature Importance Results (Permutation) ---")
        print("Features are ranked from least to most important.")
        print("A higher score means the feature has a bigger impact on the model's decisions.")
        print(perm_df)
        print("--------------------------------------------------")

        # Create and save a plot
        plt.figure(figsize=(10, 8))
        plt.barh(perm_df.index, perm_df['Importance'])
        plt.xlabel("Importance Score")
        plt.title("Permutation Feature Importance for SAC Agent")
        plt.tight_layout()
        plot_path = paths.RESULTS_DIR / "feature_importance.png"
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"\nFeature importance plot saved to: {plot_path}")
        logger.info("="*80)
        logger.info("✅ Feature analysis completed successfully.")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"An error occurred during feature analysis: {e}", exc_info=True)

if __name__ == "__main__":
    analyze_model_features()