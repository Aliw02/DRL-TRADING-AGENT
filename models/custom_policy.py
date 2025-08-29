import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils.logger import get_logger
from config.init import config

logger = get_logger(__name__)

class CustomActorCriticPolicy(BaseFeaturesExtractor):
    """Dual-stream policy network with specialized CE signal processing"""
    
    def __init__(self, observation_space, features_dim=None):
        # Load configuration
        features_dim = features_dim or config.get('model.features_dim', 256)
        ce_stream_hidden = config.get('model.ce_stream_hidden', 64)
        context_stream_hidden = config.get('model.context_stream_hidden', 128)
        position_stream_hidden = config.get('model.position_stream_hidden', 16)
        
        super(CustomActorCriticPolicy, self).__init__(observation_space, features_dim)
        
        self.sequence_length = observation_space.shape[0]
        self.feature_dim = observation_space.shape[1]
        
        # CE Stream (first 4 features)
        self.ce_stream = nn.LSTM(
            input_size=4, 
            hidden_size=ce_stream_hidden, 
            num_layers=1,
            batch_first=True,
            dropout=0.1
        )
        
        # Context Stream (next 12 features)
        self.context_stream = nn.LSTM(
            input_size=12, 
            hidden_size=context_stream_hidden, 
            num_layers=1,
            batch_first=True,
            dropout=0.1
        )
        
        # Position and equity stream (last 2 features)
        self.position_stream = nn.Sequential(
            nn.Linear(2, position_stream_hidden),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Calculate input size for fully connected layers
        fc_input_size = ce_stream_hidden + context_stream_hidden + position_stream_hidden
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, features_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Value and policy heads
        self.value_net = nn.Sequential(
            nn.Linear(features_dim, 1)
        )
        
        self.policy_net = nn.Sequential(
            nn.Linear(features_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Custom policy network initialized with features_dim: {features_dim}")
    
    def _initialize_weights(self):
        """Initialize weights for better training stability"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, observations):
        try:
            # Split the input into streams
            ce_features = observations[:, :, :4]    # CE features
            context_features = observations[:, :, 4:16]  # Context features
            position_features = observations[:, -1, 16:18]  # Position and equity (take only the last step)
            
            # Process CE stream
            ce_out, (ce_hn, _) = self.ce_stream(ce_features)
            ce_out = ce_hn[-1]  # Take the last hidden state
            
            # Process context stream
            context_out, (context_hn, _) = self.context_stream(context_features)
            context_out = context_hn[-1]
            
            # Process position stream
            position_out = self.position_stream(position_features)
            
            # Concatenate all streams
            combined = torch.cat([ce_out, context_out, position_out], dim=1)
            
            # Process through fully connected layers
            features = self.fc(combined)
            
            # Return only the extracted features (Stable-Baselines3 expects a single tensor)
            return features
            
        except Exception as e:
            logger.error(f"Error in policy forward pass: {e}")
            # Return safe values on error
            batch_size = observations.shape[0]
            feat_dim = getattr(self, '_features_dim', None) or getattr(self, 'features_dim', None) or 256
            features = torch.zeros(batch_size, feat_dim, device=observations.device)
            return features