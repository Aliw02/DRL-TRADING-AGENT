# main.py (UPDATED - The primary entry point for the entire workflow)
import argparse
import sys
import os

# --- Add project root to path ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logging, get_logger
from scripts.preprocess_data import run_preprocessing as run_preprocessing
from scripts.train_regime_model import train_and_analyze_regime_model
from scripts.create_enriched_dataset import enrich_dataset_with_regimes
from scripts.train_agent import run_agent_training # This now includes Walk-Forward and Fine-Tuning
from scripts.backtest_agent import run_backtest
from config.init import Config

def run_full_pipeline(config_path: str):
    """
    Orchestrates the entire, end-to-end professional workflow for the DRL agent.
    This pipeline is robust, interpretable, and follows quantitative finance best practices.
    """
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("==========================================================")
    logger.info("🚀 STARTING FULL PROFESSIONAL DRL TRADING AGENT PIPELINE 🚀")
    logger.info("==========================================================")

    try:
        # --- DATA PREPARATION STAGES ---
        logger.info("\n--- PIPELINE STEP 1: Initial Feature Engineering ---")
        run_preprocessing()

        logger.info("\n--- PIPELINE STEP 2: Dynamic Market Regime Detection ---")
        train_and_analyze_regime_model()

        logger.info("\n--- PIPELINE STEP 3: Enriching Dataset with Regime Vectors ---")
        enrich_dataset_with_regimes()

        # --- AGENT TRAINING STAGES ---
        logger.info("\n--- PIPELINE STEP 4: Professional Agent Training ---")
        # This single function now handles both Walk-Forward Training and Final Fine-Tuning
        agent_config = Config(config_path=config_path)
        run_agent_training(config=agent_config)

        # --- VALIDATION STAGE ---
        logger.info("\n--- PIPELINE STEP 5: Professional Backtesting on Unseen Data ---")
        run_backtest()
        
        logger.info("\n==========================================================")
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY! ✅")
        logger.info("==========================================================")

    except Exception as e:
        logger.error(f"A critical error occurred in the pipeline: {e}", exc_info=True)
        logger.error("Pipeline terminated prematurely.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full professional DRL agent pipeline.")
    parser.add_argument(
        '--config',
        type=str,
        default='config/config_sac.yaml',
        help="Path to the agent's training configuration file (e.g., config/config_sac.yaml)."
    )
    args = parser.parse_args()
    
    run_full_pipeline(config_path=args.config)