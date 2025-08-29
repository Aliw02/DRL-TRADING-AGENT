# scripts/analyze_features.py (NEW SCRIPT)

import pandas as pd
import joblib
import torch
import shap
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sb3_contrib import RecurrentPPO
from utils.data_transformation import DataTransformer
from sklearn.inspection import permutation_importance

# --- Configuration ---
MODEL_PATH = "results/final_model_for_live/finetuned_model.zip"
SCALER_PATH = "results/final_model_for_live/final_robust_scaler.joblib"
TEST_DATA_PATH = "data/XAUUSDM15-TEST_UNSEEN.csv" # Use unseen test data

def analyze_model_features():
    """
    Loads a trained model and runs advanced feature analysis using
    Permutation Importance and SHAP.
    """
    print("===== Starting Feature Analysis Process =====")
    
    # 1. Load Model, Scaler, and Data
    model = RecurrentPPO.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    transformer = DataTransformer()
    df = transformer.load_and_preprocess_data(file_path=TEST_DATA_PATH)
    
    # Separate features from the target ('close' price)
    feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'time']]
    X_test_unscaled = df[feature_cols]
    
    # Scale the features
    X_test_scaled = scaler.transform(X_test_unscaled)
    
    # We need to reshape the data for the model (add sequence length)
    # This is a simplified example; a real implementation would use sliding windows.
    num_samples = 500 # Use a subset for faster analysis
    sequence_length = model.observation_space.shape[0]
    
    # Create sequences
    X_test_sequences = np.array([X_test_scaled[i:i+sequence_length] for i in range(num_samples)])

    print(f"Loaded and prepared {num_samples} samples for analysis.")

    # --- 2. Permutation Feature Importance ---
    print("\n--- Calculating Permutation Feature Importance ---")
    
    # We need a scoring function. Let's use the model's value function output.
    def scoring_fn(estimator, X):
        # The model expects a PyTorch tensor
        obs = torch.tensor(X, dtype=torch.float32, device=estimator.device)
        # Get the value prediction
        values, _, _ = estimator.policy.forward_actor_critic(obs)
        return values.mean().item()

    perm_importance = permutation_importance(
        model.policy, X_test_sequences, scoring_fn, n_repeats=10, random_state=42
    )
    
    sorted_idx = perm_importance.importances_mean.argsort()
    perm_df = pd.DataFrame(
        data=perm_importance.importances_mean[sorted_idx],
        index=np.array(feature_cols)[sorted_idx],
        columns=['Importance']
    )
    
    print("Top 10 most important features (Permutation):")
    print(perm_df.tail(10))
    
    # --- 3. SHAP (SHapley Additive exPlanations) Analysis ---
    print("\n--- Calculating SHAP Values (this may take a while) ---")
    
    # SHAP needs a background dataset to represent the "absence" of a feature
    background_data = torch.tensor(X_test_sequences[:100], dtype=torch.float32, device=model.device)
    
    # The SHAP explainer needs the part of the model that extracts features
    feature_extractor = model.policy.features_extractor
    
    explainer = shap.DeepExplainer(feature_extractor, background_data)
    
    # Explain a subset of the test data
    test_subset = torch.tensor(X_test_sequences[100:110], dtype=torch.float32, device=model.device)
    shap_values = explainer.shap_values(test_subset)
    
    # SHAP for time series is complex. We average over the sequence length for a summary.
    # The shape is (samples, seq_len, features). We average over axis 1.
    shap_values_avg = np.mean(np.abs(shap_values), axis=1)

    # Create a summary plot and save it
    shap.summary_plot(shap_values_avg, features=X_test_unscaled.iloc[100:110], feature_names=feature_cols, show=False)
    plt.savefig("results/plots/shap_summary_plot.png")
    plt.close()
    
    print("\nSHAP summary plot saved to 'results/plots/shap_summary_plot.png'")
    print("===== Analysis Completed =====")


if __name__ == "__main__":
    # Make sure you have the required libraries: pip install shap
    analyze_model_features()