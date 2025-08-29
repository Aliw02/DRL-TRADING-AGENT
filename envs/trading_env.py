# models/custom_policy.py (ADVANCED & PRACTICAL VERSION)

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils.logger import get_logger
from config.init import config

logger = get_logger(__name__)

class CustomActorCriticPolicy(BaseFeaturesExtractor):
    """
    A hybrid CNN-Transformer feature extractor for robust time-series analysis.
    - The CNN layer captures local, short-term patterns (like price action).
    - The Transformer layer analyzes the global context and relationships between these patterns.
    """
    
    def __init__(self, observation_space, features_dim=None):
        features_dim = features_dim or config.get('model.features_dim', 256)
        super(CustomActorCriticPolicy, self).__init__(observation_space, features_dim)
        
        self.feature_dim = observation_space.shape[1]

        # --- HYBRID ARCHITECTURE CONFIGURATION ---
        cnn_out_channels = config.get('model.cnn_out_channels', 64)
        transformer_n_heads = config.get('model.transformer_n_heads', 4)
        transformer_n_layers = config.get('model.transformer_n_layers', 2)
        
        # 1. CNN Layer for local feature extraction
        self.cnn_extractor = nn.Sequential(
            # Input shape: (Batch, Seq_Len, Features) -> Permute to (Batch, Features, Seq_Len) for Conv1d
            nn.Conv1d(in_channels=self.feature_dim, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_out_channels, out_channels=features_dim, kernel_size=3, padding=1),
            nn.ReLU()
            # Output shape: (Batch, features_dim, Seq_Len) -> Permute back to (Batch, Seq_Len, features_dim)
        )

        # 2. Positional Encoding for the Transformer
        self.positional_encoding = nn.Parameter(torch.zeros(1, observation_space.shape[0], features_dim))

        # 3. Transformer Encoder for global context analysis
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features_dim, 
            nhead=transformer_n_heads, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_n_layers)
        
        # 4. Final layer to process the Transformer's output
        # This will be the input to the Actor and Critic networks in SB3
        self.final_layer = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
        logger.info(f"Hybrid CNN-Transformer policy initialized with features_dim: {features_dim}")
    
    def forward(self, observations):
        # observations shape: (batch_size, sequence_length, feature_dim)
        
        # 1. Pass through CNN
        # Permute to fit Conv1d input requirement: (Batch, Features, Seq_Len)
        x = observations.permute(0, 2, 1)
        cnn_features = self.cnn_extractor(x)
        # Permute back: (Batch, Seq_Len, Features)
        x = cnn_features.permute(0, 2, 1)
        
        # 2. Add positional encoding
        x = x + self.positional_encoding
        
        # 3. Pass through Transformer
        transformer_output = self.transformer_encoder(x)
        
        # 4. We take the output of the last time step, which summarizes the sequence
        final_features = transformer_output[:, -1, :]
        
        # 5. Process through the final linear layer
        extracted_features = self.final_layer(final_features)
        
        return extracted_features