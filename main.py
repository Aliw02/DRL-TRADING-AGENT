# main.py
# FINAL, HIERARCHICAL CENTRAL COMMAND SCRIPT

import argparse
import sys
import os

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logging, get_logger
from scripts.preprocess_data import run_preprocessing
from scripts.train_regime_model import train_and_analyze_regime_model
from scripts.create_enriched_dataset import enrich_dataset_with_regimes
from scripts.train_specialists import run_specialist_training_pipeline
from scripts.backtest_hierarchical import run_hierarchical_backtest
from scripts.plot_results import Plotter
from config.init import Config
from config import paths

def run_full_hierarchical_pipeline(config_path: str):
    """
    Executes the entire DRL agent development pipeline, from raw data to a
    production-ready hierarchical squad of specialist agents.
    """
    setup_logging()
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("🚀 INITIATING HIERARCHICAL AGENT DEVELOPMENT PIPELINE 🚀")
    logger.info("=" * 60)

    try:
        # --- STAGE 1: Foundational Data Engineering ---
        logger.info("\n--- PIPELINE STAGE 1: Initial Feature Engineering ---")
        if not os.path.exists(paths.PROCESSED_DATA_FILE):
            run_preprocessing()
        else:
            logger.info(f"Preprocessed data already exists at: {paths.PROCESSED_DATA_FILE}. Skipping preprocessing.")

        # --- STAGE 2: Dynamic Market Regime Detection (Command Unit Training) ---
        logger.info("\n--- PIPELINE STAGE 2: Command Unit Training (Regime Detection) ---")
        if not os.path.exists(paths.FINAL_MODEL_DIR / "regime_gmm_model.joblib") and not os.path.exists(paths.FINAL_MODEL_DIR / "regime_robust_scaler.joblib"):
            train_and_analyze_regime_model()
        else:
            logger.info("Regime model and scaler already exist. Skipping regime training.")

        # --- STAGE 3: Enriching Dataset with Intelligence ---
        logger.info("\n--- PIPELINE STAGE 3: Enriching Dataset with Regime Intelligence ---")
        if not os.path.exists(paths.FINAL_ENRICHED_DATA_FILE):
            enrich_dataset_with_regimes()
        else:
            logger.info(f"Enriched dataset already exists at: {paths.FINAL_ENRICHED_DATA_FILE}. Skipping enrichment.")

        # --- STAGE 4: Forge the Specialist Squad ---
        logger.info("\n--- PIPELINE STAGE 4: Forging Specialist Agent Squad ---")
        run_specialist_training_pipeline(config_path=config_path)
        
        # --- STAGE 5: Full Hierarchical Combat Simulation ---
        logger.info("\n--- PIPELINE STAGE 5: Hierarchical Combat Simulation ---")
        run_hierarchical_backtest()
        
        # --- STAGE 6: Visualization of Hierarchical Performance ---
        logger.info("\n--- PIPELINE STAGE 6: Generating Strategic Visualizations ---")
        # Note: The plotter will need to be pointed to the new hierarchical result files
        plotter = Plotter() 
        plotter.run_all_plots()

        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL HIERARCHICAL PIPELINE STAGES COMPLETED SUCCESSFULLY ✅")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"A critical failure occurred in the pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full professional DRL agent pipeline.")
    parser.add_argument('--config', type=str, default='config/config_sac.yaml', help="Path to the agent's training config.")
    args = parser.parse_args()
    
    run_full_hierarchical_pipeline(config_path=args.config)