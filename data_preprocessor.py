# data_preprocessor.py (FINAL & CORRECTED VERSION)
import pandas as pd
import joblib
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import paths
from config.init import config # <-- **استيراد الإعدادات**
from utils.logger import get_logger

logger = get_logger(__name__)

class LiveDataPreprocessor:
    def __init__(self):
        # **الحل هنا: نقوم بتخزين طول التسلسل الصحيح عند تهيئة الكلاس**
        self.sequence_length = config.get('environment.sequence_length', 150)
        self.agent_scaler = None
        self.regime_model = None
        self.regime_scaler = None
        self._load_artifacts()
        logger.info(f"Preprocessor initialized. Expecting sequence length of {self.sequence_length}.")

    def _load_artifacts(self):
        try:
            logger.info("Loading all required production artifacts...")
            self.agent_scaler = joblib.load(str(paths.FINAL_SCALER_PATH))
            regime_model_path = paths.FINAL_MODEL_DIR / "regime_gmm_model.joblib"
            self.regime_model = joblib.load(regime_model_path)
            regime_scaler_path = paths.FINAL_MODEL_DIR / "regime_robust_scaler.joblib"
            self.regime_scaler = joblib.load(regime_scaler_path)
        except FileNotFoundError as e:
            logger.error(f"A required artifact was not found: {e}", exc_info=True)
            raise

    def preprocess(self, df: pd.DataFrame):
        if self.agent_scaler is None or self.regime_model is None or self.regime_scaler is None:
            logger.error("Preprocessor not initialized correctly.")
            return None

        try:
            # --- STAGE 1: Enrich with Market Regime Probabilities ---
            regime_features = ['adx', 'bb_width', 'roc_norm', 'rsi_x_adx', 'atr']
            X_raw_regime = df[regime_features].ffill().bfill()
            X_scaled_regime = self.regime_scaler.transform(X_raw_regime)
            regime_probabilities = self.regime_model.predict_proba(X_scaled_regime)
            prob_cols = [f'regime_prob_{i}' for i in range(self.regime_model.n_components)]
            prob_df = pd.DataFrame(regime_probabilities, index=df.index, columns=prob_cols)
            enriched_df = df.join(prob_df)

            # --- STAGE 2: Scale the Final Feature Set for the Agent ---
            feature_cols = [col for col in enriched_df.columns if col not in ['open', 'high', 'low', 'close', 'time', 'timestamp']]
            scaled_features = self.agent_scaler.transform(enriched_df[feature_cols])

            # --- STAGE 3: Prepare the Observation Sequence ---
            if len(scaled_features) < self.sequence_length:
                logger.warning(f"Not enough data for a full sequence. Need {self.sequence_length}, have {len(scaled_features)}.")
                return None

            # نأخذ آخر `sequence_length` من البيانات
            sequence = scaled_features[-self.sequence_length:]

            # النموذج يتوقع وجود مكان لخاصية "حجم الصفقة"
            position_placeholder = np.zeros((sequence.shape[0], 1), dtype=np.float32)
            observation = np.concatenate([sequence, position_placeholder], axis=1).astype(np.float32)
            final_observation = np.expand_dims(observation, axis=0)
            
            # **فحص أخير للتأكد من أن الأبعاد صحيحة**
            if final_observation.shape != (1, self.sequence_length, len(feature_cols) + 1):
                 logger.error(f"CRITICAL SHAPE MISMATCH! Final shape is {final_observation.shape}, but should be {(1, self.sequence_length, len(feature_cols) + 1)}")
                 return None

            return final_observation

        except Exception as e:
            logger.error(f"An error occurred during live data preprocessing: {e}", exc_info=True)
            return None