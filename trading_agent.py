# trading_agent.py
from sb3_contrib import RecurrentPPO
import os

# تحديد مسار النموذج النهائي
FINAL_MODEL_PATH = "results/final_model_for_live/finetuned_model.zip"

def load_trading_model():
    """
    يقوم بتحميل النموذج الذي تم تدريبه مسبقاً.
    """
    try:
        if not os.path.exists(FINAL_MODEL_PATH):
            print(f"Model file not found at: {FINAL_MODEL_PATH}")
            return None
            
        model = RecurrentPPO.load(FINAL_MODEL_PATH)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error occurred while loading model: {e}")
        return None

def get_action_from_model(model, observation, states):
    """
    يطلب من النموذج اتخاذ إجراء بناءً على الملاحظة الحالية.
    """
    try:
        action, states = model.predict(observation, state=states, deterministic=True)
        return action, states
    except Exception as e:
        print(f"Failed to predict action: {e}")
        return None