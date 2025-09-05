# models/custom_policy.py
# FINAL, HIGH-EFFICIENCY, CNN-BASED POLICY ARCHITECTURE

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils.logger import get_logger
from config.init import config

logger = get_logger(__name__)

class HybridCNNTransformerPolicy(BaseFeaturesExtractor):
    """
    A high-throughput feature extractor optimized for speed and performance.
    It uses a deep 1D Convolutional Neural Network (CNN) to extract salient
    temporal features, followed by an adaptive pooling layer to create a
    fixed-size representation for the decision-making head.

    This architecture is significantly faster than a full Transformer while
    maintaining excellent pattern recognition capabilities.
    """
    
    def __init__(self, observation_space, features_dim: int = 256):
        features_dim = config.get('model.features_dim', 256)
        super().__init__(observation_space, features_dim)
        
        self.feature_dim = observation_space.shape[1]

        # --- Deep CNN Stream for Hierarchical Feature Extraction ---
        cnn_channels = config.get('model.cnn_out_channels', 64)
        self.cnn_stream = nn.Sequential(
            # Input shape: (batch, features, sequence_length)
            nn.Conv1d(in_channels=self.feature_dim, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.GELU(),
            nn.Conv1d(in_channels=cnn_channels * 2, out_channels=cnn_channels * 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels * 4),
            nn.GELU(),
        )
        
        # --- Adaptive Pooling Layer ---
        # This layer takes the output of the CNN (which has a variable sequence length)
        # and creates a single, fixed-size feature vector. This is more robust
        # than simply taking the last timestep.
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # --- Final MLP Head for Decision Making ---
        self.fc_head = nn.Sequential(
            nn.Linear(cnn_channels * 4, features_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(features_dim, features_dim)
        )
        
        logger.info(f"✅ High-Efficiency CNN policy initialized.")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, sequence_length, feature_dim)
        # Permute for CNN: (batch_size, feature_dim, sequence_length)
        x = observations.permute(0, 2, 1)

        # 1. Pass through CNN stream
        cnn_output = self.cnn_stream(x)
        
        # 2. Pool the features across the time dimension
        pooled_output = self.pooling(cnn_output).squeeze(-1) # Squeeze the last dimension
        
        # 3. Pass through the final MLP head
        features = self.fc_head(pooled_output)
        
        return features