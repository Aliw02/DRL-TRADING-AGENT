# scripts/analyze_specialists.py
# STRATEGIC INTELLIGENCE MODULE FOR HIERARCHICAL SQUAD ANALYSIS

import pandas as pd
import joblib
import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import json

# --- Add project root to path ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.custom_policy import HybridCNNTransformerPolicy
from utils.accelerator import DEVICE
from stable_baselines3 import SAC
from utils.logger import setup_logging, get_logger
from config.init import Config
from config import paths

def analyze_single_specialist(specialist_model, data: pd.DataFrame, feature_cols: list, specialist_id: int):
    """
    Runs Permutation Feature Importance analysis on a single specialist agent.
    """
    logger = get_logger(__name__)
    logger.info(f"--- Analyzing Specialist for Regime {specialist_id} ---")

    # --- Prepare data sequences for the specialist ---
    num_samples = 2000
    sequence_length = specialist_model.policy.observation_space.shape[1]
    
    if len(data) < sequence_length + num_samples:
        logger.warning(f"Insufficient data for Regime {specialist_id} to perform analysis. Skipping.")
        return None

    X_sequences = np.array([data[feature_cols].iloc[i:i+sequence_length].values for i in range(num_samples)])
    position_placeholder = np.zeros((num_samples, sequence_length, 1))
    X_sequences_with_pos = np.concatenate([X_sequences, position_placeholder], axis=2).astype(np.float32)

    # --- Define the scoring function ---
    def scoring_fn(model, X):
        actions, _ = model.predict(X, deterministic=True)
        # Score is the magnitude of the action, indicating conviction
        return np.mean(actions**2)

    baseline_score = scoring_fn(specialist_model, X_sequences_with_pos)
    importances = []

    for i in range(len(feature_cols)):
        X_permuted = X_sequences_with_pos.copy()
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
    of each specialist agent in the hierarchical squad.
    """
    setup_logging()
    logger = get_logger(__name__)
    logger.info("="*80); logger.info(" HIERARCHICAL SQUAD INTELLIGENCE ANALYSIS "); logger.info("="*80)

    try:
        # --- 1. Load Master Artifacts and Regime-Partitioned Data ---
        full_df = pd.read_parquet(paths.FINAL_ENRICHED_DATA_FILE)
        regime_model = joblib.load(paths.FINAL_MODEL_DIR / "regime_gmm_model.joblib")
        regime_scaler = joblib.load(paths.FINAL_MODEL_DIR / "regime_robust_scaler.joblib")
        
        # We need the original `create_regime_datasets` logic to get the right data for each specialist
        from scripts.train_specialists import create_regime_datasets
        regime_datasets = create_regime_datasets(full_df, regime_model, regime_scaler)
        
        num_regimes = regime_model.n_components
        all_importances = {}

        # --- 2. Analyze Each Specialist Individually ---
        for i in range(num_regimes):
            model_path = paths.FINAL_MODEL_DIR / f"specialist_regime_{i}/models/best_model.zip"
            if not os.path.exists(model_path): continue

            specialist_model = SAC.load(model_path, device=DEVICE)
            regime_data = regime_datasets[i]
            
            feature_cols = [col for col in regime_data.columns if col not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
            
            importance_df = analyze_single_specialist(specialist_model, regime_data, feature_cols, i)
            if importance_df is not None:
                all_importances[f"Regime_{i}"] = importance_df
                
                # --- 3. Visualize and Save Individual Report ---
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

        # --- 4. Create a Consolidated Report ---
        if all_importances:
            consolidated_df = pd.concat(all_importances, axis=1)
            print("\n" + "="*80)
            print("      CONSOLIDATED STRATEGIC FEATURE IMPORTANCE REPORT")
            print("="*80)
            print(consolidated_df.head(20))
            print("="*80)
            consolidated_df.to_csv(paths.RESULTS_DIR / "consolidated_feature_importance.csv")

    except Exception as e:
        logger.error(f"A critical failure occurred during squad analysis: {e}", exc_info=True)