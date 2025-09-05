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
        
# models/custom_policy.py
# STATE-OF-THE-ART, PRODUCTION-GRADE POLICY ARCHITECTURE

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils.logger import get_logger
from config.init import config

logger = get_logger(__name__)

class HybridCNNTransformerPolicy(BaseFeaturesExtractor):
    """
    An advanced hybrid feature extractor combining:
    1. 1D Convolutional Neural Networks (CNNs) for robust local pattern detection
       (e.g., candlestick patterns, momentum bursts).
    2. A Transformer Encoder for capturing long-range temporal dependencies and market context.
    3. A dedicated [CLS] token to create a single, powerful vector representation
       of the entire input sequence for decision-making.
    """

    def __init__(self, observation_space, features_dim: int = 256):
        # The master features_dim is loaded from config, ensuring consistency.
        features_dim = config.get('model.features_dim', 256)
        super().__init__(observation_space, features_dim)

        self.sequence_length = observation_space.shape[0]
        self.feature_dim = observation_space.shape[1]

        # --- 1. CNN Stream for Local Feature Extraction ---
        # This stream acts as a sophisticated feature pre-processor for the Transformer.
        cnn_out_channels = config.get('model.cnn_out_channels', 64)
        self.cnn_stream = nn.Sequential(
            # Input shape: (batch, features, sequence_length)
            nn.Conv1d(in_channels=self.feature_dim, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels), # BatchNorm stabilizes training significantly.
            nn.GELU(), # Modern activation function, often outperforms ReLU.
            nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_out_channels * 2),
            nn.GELU(),
        )

        transformer_input_dim = cnn_out_channels * 2

        # --- 2. Learnable [CLS] Token ---
        # This token will learn to aggregate information from the entire sequence.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_input_dim))

        # --- 3. Transformer Stream for Global Context ---
        n_heads = config.get('model.transformer_n_heads', 4)
        n_layers = config.get('model.transformer_n_layers', 2)
        dropout = config.get('model.transformer_dropout', 0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_input_dim,
            nhead=n_heads,
            dropout=dropout,
            activation='gelu', # Use GELU for consistency
            batch_first=True, # Critical for correct tensor dimensions
            norm_first=True # Pre-LayerNorm for more stable training
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- 4. Final MLP Head for Decision Making ---
        # This head processes the aggregated [CLS] token output.
        self.fc_head = nn.Sequential(
            nn.LayerNorm(transformer_input_dim), # Normalize before final projection
            nn.Linear(transformer_input_dim, features_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(features_dim, features_dim)
            # No final activation, as this is the input to the actor/critic networks
        )

        logger.info(f"✅ State-of-the-art Hybrid CNN-Transformer policy initialized.")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, sequence_length, feature_dim)
        # Permute for CNN: (batch_size, feature_dim, sequence_length)
        x = observations.permute(0, 2, 1)

        # 1. Pass through CNN stream to extract local patterns
        cnn_output = self.cnn_stream(x)

        # Permute back for Transformer: (batch_size, sequence_length, cnn_output_dim)
        transformer_input = cnn_output.permute(0, 2, 1)

        # 2. Prepend the [CLS] token to the sequence
        batch_size = transformer_input.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        transformer_input_with_cls = torch.cat((cls_tokens, transformer_input), dim=1)

        # 3. Pass through Transformer Encoder to capture global context
        transformer_output = self.transformer_encoder(transformer_input_with_cls)

        # 4. Isolate the output of the [CLS] token. It now represents the entire sequence.
        cls_output = transformer_output[:, 0, :]

        # 5. Pass the aggregated representation through the final MLP head
        features = self.fc_head(cls_output)

        return features