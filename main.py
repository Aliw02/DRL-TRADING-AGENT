# trading_agent.py (NEW WORKFLOW ORCHESTRATOR)

import os
import sys
import argparse

# --- Add project root to the Python path ---
# This allows us to import modules from other directories like 'scripts' and 'utils'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
print(f"Project Root added to sys.path: {PROJECT_ROOT}")
# --- Import the core functions from our scripts ---
from scripts.train_agent import run_walk_forward_training, run_finetuning_for_live_trading
from scripts.analyze_features import analyze_model_features
from scripts.backtest_agent import run_backtest
from utils.logger import setup_logging, get_logger


def main(skip_training=False):
    """
    Orchestrates the full, end-to-end workflow for the DRL Trading Agent.
    It runs the following stages in sequence:
    1. Full Walk-Forward Training.
    2. Fine-tuning the best model for live trading.
    3. Analyzing the final model's feature importance using SHAP.
    4. Backtesting the final model on unseen data.
    """
    # Initialize our structured logging
    setup_logging()
    logger = get_logger(__name__)

    try:
        if not skip_training:
            # --- STAGE 1: TRAINING ---
            print("="*80)
            print("STAGE 1: Starting Walk-Forward Training...")
            print("This is the longest stage, where the agent learns on expanding windows of data.")
            print("="*80)
            run_walk_forward_training()
            print("✅ STAGE 1: Walk-Forward Training completed successfully.")

            # --- STAGE 2: FINE-TUNING ---
            print("="*80)
            print("STAGE 2: Starting Fine-Tuning of the Best Model...")
            print("The best model from the last split is now trained on the entire dataset.")
            print("="*80)
            run_finetuning_for_live_trading()
            print("✅ STAGE 2: Fine-Tuning completed successfully.")
        else:
            print("Skipping Training and Fine-Tuning stages as requested.")

        # --- STAGE 3: ANALYSIS ---
        print("="*80)
        print("STAGE 3: Analyzing Final Model's Feature Importance...")
        print("Using SHAP to understand what the model has learned.")
        print("="*80)
        analyze_model_features()
        print("✅ STAGE 3: Feature analysis completed successfully.")

        # --- STAGE 4: BACKTESTING ---
        print("="*80)
        print("STAGE 4: Backtesting Final Model on Unseen Data...")
        print("Simulating the agent's performance on new historical data.")
        print("="*80)
        run_backtest()
        print("✅ STAGE 4: Backtesting completed successfully.")

        print("\n\n" + "*"*30)
        print(">>> FULL WORKFLOW COMPLETED! <<<")
        print("*"*30)
        print("Your agent is trained, analyzed, and backtested. It is ready for live deployment.")

    except FileNotFoundError as e:
        print(f"A required file was not found: {e}")
        print("Please ensure your data files and model paths are correct.")
        print("Workflow terminated prematurely.")
    except Exception as e:
        print(f"An unexpected error occurred during the workflow: {e}")
        print("Workflow terminated prematurely.")

if __name__ == "__main__":
    # We can add a command-line argument to skip the lengthy training process
    # if we only want to run analysis and backtesting on an existing model.
    parser = argparse.ArgumentParser(description="Run the full DRL Trading Agent workflow.")
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help="Skip the training and fine-tuning stages and run analysis/backtesting on existing models."
    )
    args = parser.parse_args()
    
    main(skip_training=args.skip_training)