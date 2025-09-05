# live_trading/data_preprocessor.py
# LIVE DATA PREPARATION AND NORMALIZATION UNIT (HIERARCHICAL-AWARE)

import pandas as pd
import joblib
import numpy as np
import sys
import os

# --- Add project root to path for robust imports ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import paths
from config.init import config
from utils.logger import get_logger

logger = get_logger(__name__)

class LiveDataPreprocessor:
    def __init__(self):
        """
        Initializes the preprocessor for the hierarchical squad. It loads all
        necessary artifacts for both regime classification and specialist observation.
        """
        self.sequence_length = config.get('environment.sequence_length', 150)
        self.agent_scaler = None
        self.regime_model = None
        self.regime_scaler = None
        self._load_artifacts()
        logger.info(f"Live Preprocessor initialized. Expecting sequence length of {self.sequence_length}.")

    def _load_artifacts(self):
        try:
            logger.info("Loading all required production artifacts for preprocessing...")
            # This is the scaler for the specialist agents' features
            self.agent_scaler = joblib.load(str(paths.FINAL_SCALER_PATH))
            # These are for the Command Unit (regime classification)
            self.regime_model = joblib.load(paths.FINAL_MODEL_DIR / "regime_gmm_model.joblib")
            self.regime_scaler = joblib.load(paths.FINAL_MODEL_DIR / "regime_robust_scaler.joblib")
        except FileNotFoundError as e:
            logger.error(f"A required artifact was not found: {e}", exc_info=True)
            raise

    def preprocess_for_specialist(self, df: pd.DataFrame):
        """
        Prepares the final observation vector specifically for a specialist agent.
        It ensures that regime probability features are EXCLUDED, matching the
        specialist's training data.
        """
        if not self.agent_scaler:
            logger.error("Preprocessor not initialized correctly. Agent scaler is missing.")
            return None
        try:
            # --- STAGE 1: Scale the Feature Set for the Specialist ---
            # The specialist was trained on data without regime probabilities.
            feature_cols = [col for col in df.columns if 'regime_prob' not in col and col not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
            scaled_features = self.agent_scaler.transform(df[feature_cols])

            # --- STAGE 2: Prepare the Observation Sequence ---
            if len(scaled_features) < self.sequence_length:
                logger.warning(f"Not enough data for a full sequence. Need {self.sequence_length}, have {len(scaled_features)}.")
                return None

            sequence = scaled_features[-self.sequence_length:]
            # The agent expects a placeholder for the current position size
            position_placeholder = np.zeros((sequence.shape[0], 1), dtype=np.float32)
            observation = np.concatenate([sequence, position_placeholder], axis=1).astype(np.float32)
            
            # Final observation shape must be (1, sequence_length, num_features + 1)
            final_observation = np.expand_dims(observation, axis=0)
            
            return final_observation
        except Exception as e:
            logger.error(f"An error occurred during live data preprocessing for specialist: {e}", exc_info=True)
            return None