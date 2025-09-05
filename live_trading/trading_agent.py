# live_trading/trading_agent.py
# HIERARCHICAL SQUAD ASSEMBLY AND INFERENCE UNIT

import os
import sys
import torch
import joblib
import json
from stable_baselines3 import SAC

# --- Add project root to path for robust imports ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import paths
from utils.accelerator import DEVICE
from models.custom_policy import HybridCNNTransformerPolicy # Used by specialists
from utils.logger import get_logger

def load_specialist_squad():
    """
    Assembles the entire hierarchical squad:
    1. The Command Unit (Regime Classifier + Scaler).
    2. All battle-ready Specialist Agents.
    Returns a dictionary containing the full squad, or None on failure.
    """
    logger = get_logger(__name__)
    squad = {'specialists': {}}
    try:
        logger.info("Assembling specialist squad for live operations...")
        logger.info("Loading Command Unit (Regime Classifier and Scaler)...")
        squad['classifier'] = joblib.load(paths.FINAL_MODEL_DIR / "regime_gmm_model.joblib")
        squad['scaler'] = joblib.load(paths.FINAL_MODEL_DIR / "regime_robust_scaler.joblib")
        
        with open(paths.FINAL_MODEL_DIR / "regime_config.json", 'r') as f:
            num_regimes = json.load(f)['optimal_n_clusters']
            
        logger.info(f"Loading {num_regimes} specialist agents from the arsenal...")
        for i in range(num_regimes):
            model_path = paths.FINAL_MODEL_DIR / f"specialist_regime_{i}/models/best_model.zip"
            if os.path.exists(model_path):
                # Specialists are loaded to CPU for inference to conserve GPU memory if needed
                squad['specialists'][i] = SAC.load(model_path, device='cpu')
                logger.info(f"-> Specialist for Regime {i} is operational.")
            else:
                logger.warning(f"Model for specialist {i} not found. This regime will be untradable.")
        
        logger.info("✅ Specialist squad fully assembled and ready for deployment.")
        return squad
    except Exception as e:
        logger.error(f"Critical failure during squad assembly: {e}", exc_info=True)
        return None

def get_specialist_action_and_uncertainty(specialist_model, observation):
    """
    Gets a trading action and its uncertainty from a specific specialist model.
    """
    try:
        obs_tensor = torch.as_tensor(observation).float().to(specialist_model.device)
        mean_action, log_std = specialist_model.actor.get_dist_params(obs_tensor)
        std = torch.exp(log_std)
        return mean_action.detach().cpu().numpy(), std.detach().cpu().numpy()
    except Exception as e:
        return None, None