# trading_agent.py (UPDATED for SAC and new paths)
from stable_baselines3 import SAC
import os
from config import paths # Import the centralized paths

def load_trading_model():
    """
    Loads the final, pre-trained SAC trading model.
    """
    try:
        model_path = str(paths.FINAL_MODEL_PATH)
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at: {model_path}")
            print("Please run the main training pipeline first.")
            return None
            
        # Load the SAC model
        model = SAC.load(model_path)
        print("✅ SAC Trading model loaded successfully.")
        return model
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None

def get_action_from_model(model, observation):
    """
    Gets a trading action from the model based on the current observation.
    Note: SAC is not recurrent, so it does not use 'states'.
    """
    try:
        # Get a deterministic action for live trading
        action, _ = model.predict(observation, deterministic=True)
        return action
    except Exception as e:
        print(f"Failed to predict action: {e}")
        return None