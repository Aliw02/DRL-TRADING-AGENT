# scripts/stress_tester.py
# ADVERSARIAL STRESS TESTING AND COMBAT SIMULATION UNIT

import pandas as pd
import numpy as np
import os
import sys

# --- Add project root to path for robust imports ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from utils.logger import setup_logging, get_logger
from scripts.backtest_hierarchical import run_hierarchical_backtest # We reuse the core backtester
from config import paths

def generate_flash_crash_scenario(base_data: pd.DataFrame, crash_percentage=0.20, duration=15):
    """
    Injects a synthetic flash crash event into a copy of the historical data.
    """
    logger = get_logger(__name__)
    logger.info(f"Engineering 'Flash Crash' scenario (-{crash_percentage*100}% over {duration} mins)...")
    scenario_df = base_data.copy()
    
    # Select a random point for the crash to start
    crash_start_index = np.random.randint(100, len(scenario_df) - duration - 100)
    crash_end_index = crash_start_index + duration
    
    start_price = scenario_df['close'].iloc[crash_start_index]
    crash_target_price = start_price * (1 - crash_percentage)
    
    # Create the price drop
    crash_prices = np.linspace(start_price, crash_target_price, duration)
    
    # Inject the crash into the dataframe
    scenario_df.loc[crash_start_index:crash_end_index-1, 'close'] = crash_prices
    scenario_df.loc[crash_start_index:crash_end_index-1, 'low'] = scenario_df['close'] * 0.99
    scenario_df.loc[crash_start_index:crash_end_index-1, 'open'] = np.roll(scenario_df['close'], 1)
    scenario_df.loc[crash_start_index:crash_end_index-1, 'high'] = scenario_df['open'] * 1.01
    
    return scenario_df

def generate_sustained_bear_market(base_data: pd.DataFrame, decline_percentage=0.30, duration_days=252):
    """
    Generates a long, slow, grinding bear market scenario.
    """
    logger = get_logger(__name__)
    logger.info(f"Engineering 'Sustained Bear Market' scenario (-{decline_percentage*100}% over {duration_days} days)...")
    scenario_df = base_data.copy().tail(duration_days * 24 * 4) # Use last year of data
    if len(scenario_df) < 1000: return scenario_df # Not enough data
    
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
    hierarchical squad.
    """
    setup_logging()
    logger = get_logger(__name__)
    logger.info("="*80); logger.info(" INITIATING ADVERSARIAL STRESS TESTING PROTOCOLS "); logger.info("="*80)
    
    # Load a base dataset to manipulate for scenarios
    # In a real scenario, you'd load specific historical crisis data (e.g., 2008)
    base_df = pd.read_parquet(paths.FINAL_ENRICHED_DATA_FILE)

    scenarios = {
        "flash_crash": generate_flash_crash_scenario(base_df),
        "bear_market": generate_sustained_bear_market(base_df),
        # You would add a function here to load 2008 data, etc.
        # "2008_crisis": load_crisis_data('2008') 
    }

    for name, scenario_df in scenarios.items():
        logger.info("\n" + "="*60)
        logger.info(f" RUNNING SIMULATION FOR SCENARIO: {name.upper()} ")
        logger.info("="*60)
        
        try:
            # We call the hierarchical backtester, but pass the hostile scenario data
            run_hierarchical_backtest(
                backtest_df=scenario_df, 
                results_suffix=f"_stresstest_{name}" # Save results with a special name
            )
        except Exception as e:
            logger.error(f"Stress test for '{name}' failed: {e}", exc_info=True)

if __name__ == "__main__":
    run_all_stress_tests()