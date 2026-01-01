"""
Hybrid Transformer-Autoencoder Model for Music DeepFake Detection
"""

import torch
import torch.nn as nn
from .transformer import TransformerEncoder
from .autoencoder import Autoencoder


class HybridTransformerAutoencoder(nn.Module):
    """
    Hybrid model combining Transformer encoder and Autoencoder
    for detecting synthetic music
    """
    def __init__(self, config):
        super(HybridTransformerAutoencoder, self).__init__()
        
        self.config = config
        
        # Extract feature dimensions
        # Assuming input is mel-spectrogram with n_mels frequency bins
        self.n_mels = config['data']['n_mels']
        
        # Transformer encoder for temporal modeling
        self.transformer = TransformerEncoder(
            input_dim=self.n_mels,
            d_model=config['model']['transformer']['d_model'],
            nhead=config['model']['transformer']['nhead'],
            num_layers=config['model']['transformer']['num_layers'],
            dim_feedforward=config['model']['transformer']['dim_feedforward'],
            dropout=config['model']['transformer']['dropout']
        )
        
        # Global pooling to aggregate temporal information
        self.pooling_type = 'attention'  # or 'mean', 'max'
        
        if self.pooling_type == 'attention':
            d_model = config['model']['transformer']['d_model']
            self.attention_pool = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Softmax(dim=1)
            )
        
        # Autoencoder for feature compression
        self.autoencoder = Autoencoder(
            input_dim=config['model']['transformer']['d_model'],
            encoder_dims=config['model']['autoencoder']['encoder_dims'],
            latent_dim=config['model']['autoencoder']['latent_dim'],
            decoder_dims=config['model']['autoencoder']['decoder_dims'],
            dropout=config['model']['autoencoder']['dropout']
        )
        
        # Classification head
        latent_dim = config['model']['autoencoder']['latent_dim']
        hidden_dims = config['model']['classifier']['hidden_dims']
        dropout = config['model']['classifier']['dropout']
        
        classifier_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            classifier_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (binary classification: real vs synthetic)
        classifier_layers.append(nn.Linear(prev_dim, 1))
        classifier_layers.append(nn.Sigmoid())
        
        self.classifier = nn.Sequential(*classifier_layers)
        
    def forward(self, x):
        """
        Forward pass through the hybrid model
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_mels]
               (e.g., mel-spectrogram)
        
        Returns:
            Dictionary containing:
                - logits: Classification logits
                - latent: Latent representation
                - reconstruction: Reconstructed features
        """
        # Apply transformer encoder
        # x: [batch, seq_len, n_mels]
        transformer_out = self.transformer(x)  # [batch, seq_len, d_model]
        
        # Aggregate temporal information
        if self.pooling_type == 'attention':
            # Attention-based pooling
            attention_weights = self.attention_pool(transformer_out)  # [batch, seq_len, 1]
            pooled = torch.sum(transformer_out * attention_weights, dim=1)  # [batch, d_model]
        elif self.pooling_type == 'mean':
            pooled = torch.mean(transformer_out, dim=1)  # [batch, d_model]
        elif self.pooling_type == 'max':
            pooled = torch.max(transformer_out, dim=1)[0]  # [batch, d_model]
        
        # Apply autoencoder
        latent, reconstruction = self.autoencoder(pooled)  # latent: [batch, latent_dim]
        
        # Classification
        logits = self.classifier(latent)  # [batch, 1]
        
        return {
            'logits': logits,
            'latent': latent,
            'reconstruction': reconstruction,
            'pooled_features': pooled
        }
    
    def predict(self, x):
        """
        Make prediction on input
        
        Args:
            x: Input tensor
        
        Returns:
            Binary prediction (0: real, 1: synthetic)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            predictions = (output['logits'] > 0.5).float()
        return predictions


class HybridModelWithLoss(nn.Module):
    """
    Wrapper that includes loss computation
    """
    def __init__(self, config):
        super(HybridModelWithLoss, self).__init__()
        
        self.model = HybridTransformerAutoencoder(config)
        
        # Loss functions
        self.classification_criterion = nn.BCELoss()
        self.reconstruction_criterion = nn.MSELoss()
        
        # Loss weights
        self.classification_weight = config['training']['classification_weight']
        self.reconstruction_weight = config['training']['reconstruction_weight']
    
    def forward(self, x, labels=None):
        """
        Forward pass with optional loss computation
        """
        output = self.model(x)
        
        if labels is not None:
            # Compute classification loss
            classification_loss = self.classification_criterion(
                output['logits'].squeeze(), 
                labels.float()
            )
            
            # Compute reconstruction loss
            reconstruction_loss = self.reconstruction_criterion(
                output['reconstruction'],
                output['pooled_features']
            )
            
            # Total loss
            total_loss = (
                self.classification_weight * classification_loss +
                self.reconstruction_weight * reconstruction_loss
            )
            
            output['loss'] = total_loss
            output['classification_loss'] = classification_loss
            output['reconstruction_loss'] = reconstruction_loss
        
        return output


if __name__ == "__main__":
    # Test the hybrid model
    import yaml
    
    # Load config
    with open('../../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    batch_size = 4
    seq_len = 216  # ~5 seconds of audio at 22050 Hz
    n_mels = config['data']['n_mels']
    
    model = HybridTransformerAutoencoder(config)
    x = torch.randn(batch_size, seq_len, n_mels)
    labels = torch.randint(0, 2, (batch_size,))
    
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Latent shape: {output['latent'].shape}")
    print(f"Reconstruction shape: {output['reconstruction'].shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
