# models/custom_policy.py (REPLACEMENT CODE)

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils.logger import get_logger
from config.init import config

logger = get_logger(__name__)

class CustomActorCriticPolicy(BaseFeaturesExtractor):
    """
    A feature extractor policy using a Transformer Encoder architecture.
    """
    
    def __init__(self, observation_space, features_dim=None):
        # Load configuration
        features_dim = features_dim or config.get('model.features_dim', 256)
        
        super(CustomActorCriticPolicy, self).__init__(observation_space, features_dim)
        
        self.sequence_length = observation_space.shape[0]
        self.feature_dim = observation_space.shape[1] # Number of features per timestep

        # --- Transformer Configuration ---
        n_heads = config.get('model.transformer_n_heads', 4)  # Number of attention heads
        n_layers = config.get('model.transformer_n_layers', 2) # Number of encoder layers
        dropout = config.get('model.transformer_dropout', 0.1)
        
        # 1. Input Embedding Layer
        # A linear layer to project input features to the model's dimension
        self.input_proj = nn.Linear(self.feature_dim, features_dim)

        # 2. Positional Encoding
        # To give the model information about the order of the sequence
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.sequence_length, features_dim))

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features_dim, 
            nhead=n_heads, 
            dropout=dropout,
            batch_first=True # IMPORTANT!
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers
        )
        
        # 4. Final Fully Connected Layer to output the final features
        # We will take the output of the last timestep from the transformer
        self.fc = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        logger.info(f"Transformer policy network initialized with features_dim: {features_dim}")
    
    def forward(self, observations):
        try:
            # observations shape: (batch_size, sequence_length, feature_dim)
            
            # 1. Project input features to the model's dimension
            x = self.input_proj(observations)
            
            # 2. Add positional encoding
            x = x + self.positional_encoding
            
            # 3. Pass through the Transformer Encoder
            transformer_output = self.transformer_encoder(x)
            
            # 4. We take the output corresponding to the last time step
            # This contains the encoded information of the entire sequence
            last_timestep_output = transformer_output[:, -1, :]
            
            # 5. Pass through the final FC layers to get the extracted features
            features = self.fc(last_timestep_output)
            
            return features
            
        except Exception as e:
            logger.error(f"Error in Transformer policy forward pass: {e}")
            batch_size = observations.shape[0]
            feat_dim = getattr(self, '_features_dim', None) or getattr(self, 'features_dim', None) or 256
            features = torch.zeros(batch_size, feat_dim, device=observations.device)
            return features
        

# models/custom_policy.py (CORRECTED CLASS NAME)

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils.logger import get_logger
from config.init import config

logger = get_logger(__name__)

# --- FIX: The class name is now corrected to match the import statement ---
class HybridCNNTransformerPolicy(BaseFeaturesExtractor):
    """
    An advanced hybrid feature extractor combining:
    1. CNNs for robust local pattern detection (e.g., candlestick patterns).
    2. A Transformer Encoder for capturing long-range dependencies and context.
    3. A [CLS] token to create a single, powerful representation of the entire sequence.
    """
    
    def __init__(self, observation_space, features_dim=None):
        features_dim = features_dim or config.get('model.features_dim', 256)
        super().__init__(observation_space, features_dim)
        
        self.sequence_length = observation_space.shape[0]
        self.feature_dim = observation_space.shape[1]

        # --- 1. CNN Stream for Local Feature Extraction ---
        cnn_out_channels = config.get('model.cnn_out_channels', 64)
        self.cnn_stream = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels), # BatchNorm stabilizes training
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
        )
        
        # --- 2. [CLS] Token ---
        transformer_input_dim = cnn_out_channels
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_input_dim))

        # --- 3. Transformer Stream for Global Context ---
        n_heads = config.get('model.transformer_n_heads', 4)
        n_layers = config.get('model.transformer_n_layers', 2)
        dropout = config.get('model.transformer_dropout', 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_input_dim, 
            nhead=n_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- 4. Final Fully Connected Layer (MLP Head) ---
        self.fc_head = nn.Sequential(
            nn.LayerNorm(transformer_input_dim),
            nn.Linear(transformer_input_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
        logger.info(f"✅ State-of-the-art Hybrid CNN-Transformer policy initialized.")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.permute(0, 2, 1)
        cnn_output = self.cnn_stream(x)
        
        transformer_input = cnn_output.permute(0, 2, 1)
        
        batch_size = transformer_input.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        
        transformer_input_with_cls = torch.cat((cls_tokens, transformer_input), dim=1)
        
        transformer_output = self.transformer_encoder(transformer_input_with_cls)
        
        cls_output = transformer_output[:, 0, :]
        
        features = self.fc_head(cls_output)
        
        return features