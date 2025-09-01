import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logging, get_logger
from scripts.preprocess_data import run_preprocessing
from scripts.train_regime_model import train_and_analyze_regime_model
from scripts.create_enriched_dataset import enrich_dataset_with_regimes
from scripts.train_agent import run_agent_training, run_agent_finetuning
from scripts.analyze_features import analyze_model_features
from scripts.backtest_agent import run_backtest
from config.init import Config

def run_full_pipeline(config_path: str):
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("="*58); logger.info("🚀 STARTING FULL PROFESSIONAL DRL TRADING AGENT PIPELINE 🚀"); logger.info("="*58)

    try:
        # --- CRITICAL FIX: Each step's success is required to proceed ---
        logger.info("\\n--- PIPELINE STEP 1: Initial Feature Engineering ---")
        # run_preprocessing()

        logger.info("\\n--- PIPELINE STEP 2: Dynamic Market Regime Detection ---")
        # train_and_analyze_regime_model()

        logger.info("\\n--- PIPELINE STEP 3: Enriching Dataset with Regime Vectors ---")
        # enrich_dataset_with_regimes()

        logger.info("\\n--- PIPELINE STEP 4: Professional Agent Training ---")
        run_full_training = True # Set to False to skip initial training and only do fine-tuning
        if not run_full_training:
            logger.info("⚠️ Skipping initial training, proceeding to fine-tuning only.")
            agent_config = Config(config_path=config_path)
            run_agent_finetuning(config=agent_config)
        else:
            agent_config = Config(config_path=config_path)
            run_agent_training(config=agent_config)

        logger.info("\n--- Analyze Features ---")
        # Add feature analysis code here
        analyze_model_features()
        logger.info("FEATURE ANALYSIS COMPLETED SUCCESSFULLY! ✅")


        logger.info("\\n--- PIPELINE STEP 5: Professional Backtesting on Unseen Data ---")
        run_backtest()
        
        logger.info("\\n" + "="*58); logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY! ✅"); logger.info("="*58)

    except Exception as e:
        logger.error(f"A critical error occurred in the pipeline: {e}", exc_info=True)
        logger.error("Pipeline terminated prematurely.")
        # Re-raise the exception to stop the script execution completely in Kaggle
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full professional DRL agent pipeline.")
    parser.add_argument('--config', type=str, default='config/config_sac.yaml', help="Path to the agent's training config.")
    args = parser.parse_args()
    
    run_full_pipeline(config_path=args.config)