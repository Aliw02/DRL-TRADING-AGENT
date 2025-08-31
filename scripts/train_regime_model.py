# scripts/train_regime_model.py (NEW ADVANCED SCRIPT)
import pandas as pd
import joblib
import sys
import os
import json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

# --- Add project root to path ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import paths
from utils.logger import setup_logging, get_logger

def train_and_analyze_regime_model():
    """
    An advanced workflow to:
    1. Load preprocessed data from the main Parquet file.
    2. Scale the features using a dedicated RobustScaler.
    3. Dynamically find the optimal number of market regimes using BIC.
    4. Train the final Gaussian Mixture Model (GMM).
    5. Save the trained model, its specific scaler, and the optimal cluster count.
    6. Analyze the characteristics of each discovered regime and save the analysis.
    """
    setup_logging()
    logger = get_logger(__name__)

    logger.info("="*80)
    logger.info("STARTING DYNAMIC & INTERPRETABLE MARKET REGIME PIPELINE")
    logger.info("="*80)

    try:
        # 1. Load data from the main Parquet file
        input_path = paths.PROCESSED_DATA_FILE
        if not input_path.exists():
            raise FileNotFoundError(f"Parquet data not found at {input_path}. Run preprocess_data.py first!")
        
        logger.info(f"Loading data from {input_path}")
        df = pd.read_parquet(input_path)

        # Define the core features that characterize a market's regime
        regime_features = [
            'adx',          # Trend Strength
            'bb_width',     # Volatility
            'roc_norm',     # Normalized Momentum
            'rsi_x_adx',    # Interaction of Trend and Momentum
            'atr'           # Raw Volatility
        ]
        X_raw = df[regime_features].dropna()

        # 2. Scale features with a dedicated RobustScaler
        logger.info("Scaling features using a dedicated RobustScaler for the regime model.")
        regime_scaler = RobustScaler()
        X_scaled = regime_scaler.fit_transform(X_raw)

        # 3. Dynamically find the optimal number of clusters (regimes)
        logger.info("Finding optimal number of regimes using Bayesian Information Criterion (BIC)...")
        n_components = np.arange(2, 11)
        bics = []
        for n in n_components:
            gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
            gmm.fit(X_scaled)
            bics.append(gmm.bic(X_scaled))
        
        optimal_n = n_components[np.argmin(bics)]
        logger.info(f"Optimal number of market regimes found: {optimal_n}")

        # 4. Train the final GMM with the optimal number of components
        logger.info(f"Training final GMM with {optimal_n} components...")
        final_gmm = GaussianMixture(n_components=optimal_n, random_state=42, n_init=10)
        final_gmm.fit(X_scaled)

        # 5. Save all artifacts to the specified directory
        save_dir = paths.FINAL_MODEL_DIR
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the optimal cluster count to a JSON file
        regime_config_path = save_dir / "regime_config.json"
        with open(regime_config_path, 'w') as f:
            json.dump({'optimal_n_clusters': int(optimal_n)}, f)
        logger.info(f"Regime config saved to: {regime_config_path}")

        # Save the trained GMM model
        model_path = save_dir / "regime_gmm_model.joblib"
        joblib.dump(final_gmm, model_path)
        logger.info(f"Trained GMM model saved to: {model_path}")

        # Save the dedicated scaler for the GMM
        scaler_path = save_dir / "regime_robust_scaler.joblib"
        joblib.dump(regime_scaler, scaler_path)
        logger.info(f"Dedicated regime scaler saved to: {scaler_path}")

        # 6. Analyze and interpret the discovered regimes
        logger.info("Analyzing and interpreting the discovered regimes...")
        regime_labels = final_gmm.predict(X_scaled)
        analysis_df = X_raw.copy()
        analysis_df['regime'] = regime_labels
        
        # Calculate the mean of each indicator for each regime
        regime_characteristics = analysis_df.groupby('regime').mean()
        
        # Save the analysis to a CSV file for easy interpretation
        analysis_path = save_dir / "regime_analysis.csv"
        regime_characteristics.to_csv(analysis_path)
        logger.info(f"Regime characteristics analysis saved to: {analysis_path}")
        print("\n--- Regime Analysis Summary ---")
        print(regime_characteristics)
        print("-------------------------------\n")

        logger.info("✅ Full regime detection pipeline completed successfully.")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"An error occurred during the regime pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    train_and_analyze_regime_model()

