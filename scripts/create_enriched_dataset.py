# scripts/create_enriched_dataset.py (NEW)
import pandas as pd, joblib, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import paths
from utils.logger import setup_logging, get_logger

def enrich_dataset_with_regimes():
    setup_logging(); logger = get_logger(__name__)
    logger.info("="*80); logger.info("STARTING DATASET ENRICHMENT WITH MARKET REGIMES"); logger.info("="*80)
    try:
        logger.info("Loading pre-trained regime model and scaler...")
        model_path = paths.FINAL_MODEL_DIR / "regime_gmm_model.joblib"
        scaler_path = paths.FINAL_MODEL_DIR / "regime_robust_scaler.joblib"
        if not model_path.exists() or not scaler_path.exists(): raise FileNotFoundError("Run train_regime_model.py first!")
        gmm_model, regime_scaler = joblib.load(model_path), joblib.load(scaler_path)
        input_path = paths.PROCESSED_DATA_FILE
        if not input_path.exists(): raise FileNotFoundError(f"Base data not found at {input_path}")
        logger.info(f"Loading base data from {input_path}")
        df:pd.DataFrame = pd.read_parquet(input_path)
        regime_features = ['adx', 'bb_width', 'roc_norm', 'rsi_x_adx', 'atr']
        X_raw = df[regime_features].copy().ffill().bfill()
        X_scaled = regime_scaler.transform(X_raw)
        logger.info("Predicting regime probabilities...")
        regime_probabilities = gmm_model.predict_proba(X_scaled)
        prob_cols = [f'regime_prob_{i}' for i in range(gmm_model.n_components)]
        prob_df = pd.DataFrame(regime_probabilities, index=df.index, columns=prob_cols)
        logger.info("Merging probabilities into the dataset...")
        enriched_df = df.drop(columns=['market_regime'], errors='ignore').join(prob_df)
        output_path = paths.FINAL_ENRICHED_DATA_FILE
        enriched_df.to_parquet(output_path)
        logger.info(f"✅ Dataset enriched successfully. Saved to: {output_path}"); logger.info("="*80)
    except Exception as e:
        logger.error(f"Error during dataset enrichment: {e}", exc_info=True)

if __name__ == "__main__":
    enrich_dataset_with_regimes()