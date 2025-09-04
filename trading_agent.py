# trading_agent.py (THE ACTUAL AND FINAL FIX)
from stable_baselines3 import SAC
import os
from config import paths
from utils.accelerator import DEVICE
from models.custom_policy import CustomActorCriticPolicy # <-- **IMPORT YOUR CUSTOM POLICY**
from config.init import config as agent_config
import torch

def load_trading_model():
    """
    Loads the final, pre-trained SAC model for INFERENCE.
    This version adds the missing 'use_sde': False to the policy_kwargs
    to perfectly match the saved model's architecture.
    """
    try:
        model_path = str(paths.FINAL_MODEL_PATH)
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at: {model_path}")
            print("Please run the main training pipeline first.")
            return None

        # --- الحل بإضافة السطر المفقود هنا ---
        policy_kwargs = dict(
            features_extractor_class=CustomActorCriticPolicy,
            features_extractor_kwargs=dict(features_dim=agent_config.get('model.features_dim')),
            # This line is added to match the saved model's parameters EXACTLY.
            use_sde=False 
        )

        model = SAC.load(
            model_path,
            device=DEVICE,
            buffer_size=0,
            policy_kwargs=policy_kwargs
        )
        
        print(f"✅ SAC Trading model loaded successfully with all parameters matched.")
        return model
        
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None
    
def get_action_from_model(model, observation):
    """
    Gets a trading action from the model based on the current observation.
    """
    try:
        action, _ = model.predict(observation, deterministic=True)
        return action
    except Exception as e:
        print(f"Failed to predict action: {e}")
        return None


import torch

def get_action_and_uncertainty(model, observation):
    """
    Gets a trading action from the SAC model AND its uncertainty level.
    Uncertainty = standard deviation of the Gaussian policy.
    """
    try:
        # حول observation إلى tensor
        obs_tensor = torch.as_tensor(observation).float().to(model.device)

        # استخرج mean و log_std باستعمال دالة actor.get_dist_params
        mean_action, log_std = model.actor.get_dist_params(obs_tensor)

        # احسب std = exp(log_std)
        std = torch.exp(log_std)

        # حول للـ numpy
        action = mean_action.detach().cpu().numpy()
        uncertainty = std.detach().cpu().numpy()

        return action, uncertainty

    except Exception as e:
        print(f"Failed to predict action and uncertainty: {e}")
        return None, None
