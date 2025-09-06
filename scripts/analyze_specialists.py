# scripts/analyze_specialists.py
# FINAL, SCIENTIFICALLY-CORRECT, AND PATH-AWARE INTELLIGENCE MODULE

import pandas as pd
import joblib
import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# --- Add project root to path ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.custom_policy import HybridCNNTransformerPolicy
from utils.accelerator import DEVICE
from stable_baselines3 import SAC
from utils.logger import setup_logging, get_logger
from config import paths

def analyze_single_specialist(specialist_model, data: pd.DataFrame, feature_cols: list, specialist_id: int):
    """
    Runs Permutation Feature Importance analysis on a single specialist agent.
    This corrected version prepares the observation shape exactly as the model expects.
    """
    logger = get_logger(__name__)
    logger.info(f"--- Analyzing Specialist for Regime {specialist_id} ---")

    num_samples = 1000 # Reduced for faster execution

    obs_space_shape = specialist_model.policy.observation_space.shape
    sequence_length = obs_space_shape[0]
    features_dim = obs_space_shape[1]
    
    logger.info(f"    Model expects sequence_length={sequence_length}, features_dim={features_dim}")
    
    if len(data) < sequence_length + num_samples:
        logger.warning(f"Insufficient data for Regime {specialist_id} to perform analysis. Skipping.")
        return None

    X_sequences = np.array([data[feature_cols].iloc[i:i+sequence_length].values for i in range(num_samples)])

    num_indicator_features = X_sequences.shape[2]
    num_placeholders = features_dim - num_indicator_features
    
    if num_placeholders < 0:
        logger.error("Mismatch in feature dimensions. Analysis might fail.")
        num_placeholders = 0

    placeholders = np.zeros((num_samples, sequence_length, num_placeholders))
    observation_batch = np.concatenate([X_sequences, placeholders], axis=2).astype(np.float32)

    def scoring_fn(model, X):
        actions, _ = model.predict(X, deterministic=True)
        return np.mean(np.abs(actions))

    baseline_score = scoring_fn(specialist_model, observation_batch)
    importances = []

    logger.info("    Calculating feature importances...")
    for i in range(len(feature_cols)):
        X_permuted = observation_batch.copy()
        X_permuted[:, :, i] = np.random.permutation(X_permuted[:, :, i])
        permuted_score = scoring_fn(specialist_model, X_permuted)
        importances.append(baseline_score - permuted_score)

    perm_df = pd.DataFrame(
        data={'Importance': importances},
        index=feature_cols
    ).sort_values(by='Importance', ascending=False)
    
    return perm_df

def run_squad_analysis():
    """
    Main pipeline to analyze and interpret the decision-making process
    of each specialist agent from the final production models.
    """
    setup_logging()
    logger = get_logger(__name__)
    logger.info("="*80); logger.info(" HIERARCHICAL SQUAD INTELLIGENCE ANALYSIS "); logger.info("="*80)

    try:
        # --- FIX: Load models directly from the final model directory ---
        model_dir = paths.FINAL_MODEL_DIR
        logger.info(f"Analyzing final production models from: {model_dir}")

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Final model directory not found at {model_dir}. Please run the training pipeline first.")

        # Load the final GMM and Scaler
        gmm = joblib.load(model_dir / "regime_gmm_model.joblib")
        scaler = joblib.load(model_dir / "regime_robust_scaler.joblib")

        # Use the full preprocessed dataset for analysis to ensure enough data for all regimes
        full_df = pd.read_parquet(paths.PROCESSED_DATA_FILE)
        
        # Create the regime datasets using the final models
        regime_features = ['adx', 'bb_width', 'roc_norm', 'rsi_x_adx', 'atr']
        X_data_to_label = full_df[regime_features].dropna()
        data_scaled = scaler.transform(X_data_to_label)
        labels = gmm.predict(data_scaled)
        df_labeled = full_df.loc[X_data_to_label.index].copy()
        df_labeled['regime'] = labels
        
        num_regimes = gmm.n_components
        all_importances = {}

        # --- Analyze Each Specialist Individually ---
        for i in range(num_regimes):
            # Construct path to the final specialist models
            model_path = model_dir / f"specialist_regime_{i}/models/best_model.zip"
            if not os.path.exists(model_path): 
                logger.warning(f"No trained model found for specialist {i}. Skipping.")
                continue

            specialist_model = SAC.load(model_path, device=DEVICE)
            regime_data = df_labeled[df_labeled['regime'] == i]
            
            feature_cols = [col for col in regime_data.columns if col not in ['open', 'high', 'low', 'close', 'time', 'timestamp', 'regime']]
            
            importance_df = analyze_single_specialist(specialist_model, regime_data, feature_cols, i)
            if importance_df is not None:
                all_importances[f"Regime_{i}"] = importance_df
                
                # --- Visualize and Save Individual Report ---
                plt.style.use('seaborn-v0_8-whitegrid')
                fig, ax = plt.subplots(figsize=(12, 10))
                top_20 = importance_df.head(20).sort_values(by='Importance', ascending=True)
                ax.barh(top_20.index, top_20['Importance'], color='darkslateblue')
                ax.set_title(f"Feature Importance for Specialist Agent (Regime {i})", fontsize=16)
                plt.tight_layout()
                plot_path = paths.RESULTS_DIR / f"feature_importance_regime_{i}.png"
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"Analysis plot for specialist {i} saved to: {plot_path}")

        # --- Create a Consolidated Report ---
        if all_importances:
            consolidated_df = pd.concat(all_importances, axis=1)
            print("\n" + "="*80)
            print("      CONSOLIDATED STRATEGIC FEATURE IMPORTANCE REPORT")
            print("="*80)
            print(consolidated_df.head(20))
            print("="*80)
            consolidated_df.to_csv(paths.RESULTS_DIR / "consolidated_feature_importance.csv")
        else:
            logger.warning("No importance reports were generated as no specialist models were found or analyzed.")

    except Exception as e:
        logger.error(f"A critical failure occurred during squad analysis: {e}", exc_info=True)

if __name__ == "__main__":
    run_squad_analysis()