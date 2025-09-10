# main.py
# CE-ALIGNED HIERARCHICAL CENTRAL COMMAND SCRIPT (WITH FINE-TUNING STAGE)

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logging, get_logger
from scripts.preprocess_data import run_preprocessing
from scripts.train_specialists import run_specialist_training_pipeline, run_fine_tuning_on_specialists # Import the new function
from scripts.backtest_hierarchical import run_hierarchical_backtest
from scripts.plot_results import HierarchicalPlotter
from config import paths
from config.init import Config

def run_ce_aligned_pipeline(config_path: str):
    """
    Executes the streamlined DRL pipeline, now including a fine-tuning stage.
    """
    setup_logging()
    logger = get_logger(__name__)
    agent_config = Config(config_path=config_path)

    logger.info("=" * 60)
    logger.info("🚀 INITIATING CE-ALIGNED AGENT DEVELOPMENT PIPELINE 🚀")
    logger.info("=" * 60)

    try:
        # STAGE 1: Foundational Data Engineering
        logger.info("\n--- PIPELINE STAGE 1: Initial Feature Engineering ---")
        if not os.path.exists(paths.PROCESSED_DATA_FILE):
            run_preprocessing()
        else:
            logger.info(f"Processed data already exists at: {paths.PROCESSED_DATA_FILE}. Skipping.")

        # STAGE 2: Forge the Specialist Squad
        logger.info("\n--- PIPELINE STAGE 2: Forging CE-Aligned Specialist Squad ---")
        run_specialist_training_pipeline(config_path=config_path)
        
        # =======================================================================
        # ========== بداية التعديل: إضافة مرحلة الصقل والتحسين ==========
        # =======================================================================
        # STAGE 3: Fine-Tune Champion Specialists on Recent Data
        logger.info("\n--- PIPELINE STAGE 3: Fine-Tuning Specialists on Recent Data ---")
        run_fine_tuning_on_specialists(agent_config)
        # =======================================================================
        # ========== نهاية التعديل ==========
        # =======================================================================

        # STAGE 4: Full Hierarchical Combat Simulation
        logger.info("\n--- PIPELINE STAGE 4: Hierarchical Combat Simulation ---")
        run_hierarchical_backtest()
        
        # STAGE 5: Visualization of Performance
        logger.info("\n--- PIPELINE STAGE 5: Generating Strategic Visualizations ---")
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