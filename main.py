# main.py
# CE-ALIGNED HIERARCHICAL CENTRAL COMMAND SCRIPT

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logging, get_logger
from scripts.preprocess_data import run_preprocessing
from scripts.train_specialists import run_specialist_training_pipeline
from scripts.backtest_hierarchical import run_hierarchical_backtest
from scripts.plot_results import HierarchicalPlotter
from config import paths

def run_ce_aligned_pipeline(config_path: str):
    """
    Executes the streamlined DRL pipeline to develop a squad of CE-aligned
    specialist agents.
    """
    setup_logging()
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("🚀 INITIATING CE-ALIGNED AGENT DEVELOPMENT PIPELINE 🚀")
    logger.info("=" * 60)

    try:
        # STAGE 1: Foundational Data Engineering (Unchanged)
        logger.info("\n--- PIPELINE STAGE 1: Initial Feature Engineering ---")
        if not os.path.exists(paths.PROCESSED_DATA_FILE):
            run_preprocessing()
        else:
            logger.info(f"Processed data already exists at: {paths.PROCESSED_DATA_FILE}. Skipping.")

        # STAGE 2: Forge the Specialist Squad (New Logic)
        logger.info("\n--- PIPELINE STAGE 2: Forging CE-Aligned Specialist Squad ---")
        run_specialist_training_pipeline(config_path=config_path)
        
        # STAGE 3: Full Hierarchical Combat Simulation (New Logic)
        logger.info("\n--- PIPELINE STAGE 3: Hierarchical Combat Simulation ---")
        run_hierarchical_backtest()
        
        # STAGE 4: Visualization of Performance (Unchanged)
        logger.info("\n--- PIPELINE STAGE 4: Generating Strategic Visualizations ---")
        plotter = HierarchicalPlotter()
        plotter.run_all_plots()

        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL CE-ALIGNED PIPELINE STAGES COMPLETED SUCCESSFULLY ✅")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"A critical failure occurred in the pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DRL agent pipeline.")
    parser.add_argument('--config', type=str, default='config/config_sac.yaml', help="Path to the agent's training config.")
    args = parser.parse_args()
    
    run_ce_aligned_pipeline(config_path=args.config)