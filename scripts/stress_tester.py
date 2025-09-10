# scripts/stress_tester.py
# FINAL, ROBUST VERSION FOR TESTING FINE-TUNED MODELS

import pandas as pd
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from utils.logger import setup_logging, get_logger
from scripts.backtest_hierarchical import run_hierarchical_backtest
from utils.data_transformation import DataTransformer
from config import paths

def generate_flash_crash_scenario(base_data: pd.DataFrame, crash_percentage=0.30, duration=60):
    """
    Injects a more severe synthetic flash crash into a copy of the data.
    """
    logger = get_logger(__name__)
    logger.info(f"Engineering 'Flash Crash' scenario (-{crash_percentage*100}% over {duration} bars)...")
    scenario_df = base_data.copy()
    
    if len(scenario_df) < 200 + duration:
        logger.warning("Not enough data to generate flash crash. Skipping.")
        return scenario_df

    crash_start_index = np.random.randint(100, len(scenario_df) - duration - 100)
    crash_end_index = crash_start_index + duration
    
    start_price = scenario_df['close'].iloc[crash_start_index]
    crash_target_price = start_price * (1 - crash_percentage)
    
    # Create a sharp price drop
    crash_prices = np.linspace(start_price, crash_target_price, duration)
    
    # Inject the crash into the dataframe
    scenario_df.iloc[crash_start_index:crash_end_index, scenario_df.columns.get_loc('close')] = crash_prices
    scenario_df.iloc[crash_start_index:crash_end_index, scenario_df.columns.get_loc('low')] = scenario_df.iloc[crash_start_index:crash_end_index]['close'] * 0.99
    scenario_df.iloc[crash_start_index:crash_end_index, scenario_df.columns.get_loc('open')] = np.roll(scenario_df.iloc[crash_start_index:crash_end_index]['close'], 1)
    scenario_df.iloc[crash_start_index, scenario_df.columns.get_loc('open')] = start_price # Fix first open
    scenario_df.iloc[crash_start_index:crash_end_index, scenario_df.columns.get_loc('high')] = scenario_df.iloc[crash_start_index:crash_end_index]['open'] * 1.01
    
    return scenario_df

def generate_sustained_bear_market(base_data: pd.DataFrame, decline_percentage=0.40, duration_bars=4000):
    """
    Generates a long, grinding bear market scenario.
    """
    logger = get_logger(__name__)
    logger.info(f"Engineering 'Sustained Bear Market' scenario (-{decline_percentage*100}% over {duration_bars} bars)...")
    
    if len(base_data) < duration_bars:
        logger.warning(f"Not enough data for bear market scenario. Using all {len(base_data)} bars.")
        duration_bars = len(base_data)
        
    scenario_df = base_data.copy().tail(duration_bars)
    
    start_price = scenario_df['close'].iloc[0]
    end_price = start_price * (1 - decline_percentage)
    
    # Create a slow, downward geometric decay
    decay_rate = (end_price / start_price) ** (1 / len(scenario_df))
    trend = start_price * (decay_rate ** np.arange(len(scenario_df)))
    
    # Add noise to make it realistic
    noise = np.random.normal(0, 0.001, len(scenario_df))
    scenario_df['close'] = trend * (1 + noise)
    
    return scenario_df

def run_all_stress_tests():
    """
    Main orchestrator for running all stress test scenarios against the
    final fine-tuned hierarchical squad.
    """
    setup_logging()
    logger = get_logger(__name__)
    logger.info("="*80); logger.info(" INITIATING ADVERSARIAL STRESS TESTING PROTOCOLS "); logger.info("="*80)
    
    # --- Load and process the UNSEEN test data to use as a base ---
    logger.info("Loading and processing base data for stress scenarios...")
    try:
        transformer = DataTransformer()
        # We use the standard backtest file as the base for our scenarios
        base_df = transformer.load_and_preprocess_data(file_path=str(paths.BACKTEST_DATA_FILE), timeframe="15min")
    except Exception as e:
        logger.error(f"Failed to load base data for stress testing: {e}", exc_info=True)
        return

    scenarios = {
        "flash_crash": generate_flash_crash_scenario(base_df),
        "bear_market": generate_sustained_bear_market(base_df),
    }

    for name, scenario_df in scenarios.items():
        logger.info("\n" + "="*60)
        logger.info(f" RUNNING SIMULATION FOR SCENARIO: {name.upper()} ")
        logger.info("="*60)
        
        try:
            # We call the hierarchical backtester, passing the hostile scenario data directly
            run_hierarchical_backtest(
                backtest_df=scenario_df, 
                results_suffix=f"_stresstest_{name}" # Save results with a special name
            )
        except Exception as e:
            logger.error(f"Stress test for '{name}' failed: {e}", exc_info=True)

if __name__ == "__main__":
    run_all_stress_tests()