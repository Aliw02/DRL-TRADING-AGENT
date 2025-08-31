# scripts/create_enriched_dataset.py (NEW FILE)
import pandas as pd
import joblib
import sys
import os

# --- Add project root to path ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import paths
from utils.logger import setup_logging, get_logger

def enrich_dataset_with_regimes():
    """
    Loads the base preprocessed data and enriches it with market regime
    features (probability vectors) using the pre-trained GMM.
    Saves the final, enriched dataset to a new Parquet file.
    """
    setup_logging()
    logger = get_logger(__name__)

    logger.info("="*80)
    logger.info("STARTING DATASET ENRICHMENT WITH MARKET REGIMES")
    logger.info("="*80)

    try:
        # --- 1. Load required artifacts ---
        logger.info("Loading pre-trained regime model and scaler...")
        model_path = paths.FINAL_MODEL_DIR / "regime_gmm_model.joblib"
        scaler_path = paths.FINAL_MODEL_DIR / "regime_robust_scaler.joblib"
        
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError("Regime model/scaler not found. Run train_regime_model.py first!")
            
        gmm_model = joblib.load(model_path)
        regime_scaler = joblib.load(scaler_path)

        # --- 2. Load the base preprocessed data ---
        input_path = paths.DATA_DIR / "processed_training_data.parquet"
        if not input_path.exists():
            raise FileNotFoundError(f"Base data not found at {input_path}")
        
        logger.info(f"Loading base data from {input_path}")
        df = pd.read_parquet(input_path)

        # --- 3. Prepare data for prediction ---
        regime_features = [
            'adx', 'bb_width', 'roc_norm', 'rsi_x_adx', 'atr'
        ]
        
        # Handle potential NaNs robustly before scaling
        X_raw = df[regime_features].copy()
        X_raw.ffill(inplace=True)
        X_raw.bfill(inplace=True)
        
        X_scaled = regime_scaler.transform(X_raw)

        # --- 4. Get regime probabilities (the "Vector") ---
        logger.info("Predicting regime probabilities for each data point...")
        regime_probabilities = gmm_model.predict_proba(X_scaled)

        # Create a new DataFrame for the probability features
        n_clusters = gmm_model.n_components
        prob_cols = [f'regime_prob_{i}' for i in range(n_clusters)]
        prob_df = pd.DataFrame(regime_probabilities, index=df.index, columns=prob_cols)

        # --- 5. Merge with original data and save ---
        logger.info("Merging regime probabilities into the main dataset...")
        # Drop old single regime column if it exists, and merge the new probability vectors
        enriched_df = df.drop(columns=['market_regime'], errors='ignore').join(prob_df)

        # Save the final, powerful dataset
        output_path = paths.PROCESSED_DATA_FILE / "final_enriched_data.parquet"
        enriched_df.to_parquet(output_path)

        logger.info(f"✅ Dataset enriched successfully. Final shape: {enriched_df.shape}")
        logger.info(f"Enriched data saved to: {output_path}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"An error occurred during dataset enrichment: {e}", exc_info=True)

if __name__ == "__main__":
    enrich_dataset_with_regimes()
