"""
Autoencoder Module for Feature Compression and Reconstruction
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder network that compresses input features to latent space
    """
    def __init__(self, input_dim, encoder_dims, latent_dim, dropout=0.2):
        super(Encoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build encoder layers
        for dim in encoder_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Latent layer
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, input_dim]
        Returns:
            Latent representation of shape [batch_size, latent_dim]
        """
        return self.encoder(x)


class Decoder(nn.Module):
    """
    Decoder network that reconstructs features from latent space
    """
    def __init__(self, latent_dim, decoder_dims, output_dim, dropout=0.2):
        super(Decoder, self).__init__()
        
        layers = []
        prev_dim = latent_dim
        
        # Build decoder layers
        for dim in decoder_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # Reconstruct normalized features
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: Latent tensor of shape [batch_size, latent_dim]
        Returns:
            Reconstructed features of shape [batch_size, output_dim]
        """
        return self.decoder(x)


class Autoencoder(nn.Module):
    """
    Complete Autoencoder for feature learning and reconstruction
    """
    def __init__(self, input_dim, encoder_dims=[512, 256, 128], 
                 latent_dim=64, decoder_dims=[128, 256, 512], dropout=0.2):
        super(Autoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = Encoder(
            input_dim=input_dim,
            encoder_dims=encoder_dims,
            latent_dim=latent_dim,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = Decoder(
            latent_dim=latent_dim,
            decoder_dims=decoder_dims,
            output_dim=input_dim,
            dropout=dropout
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, input_dim]
        Returns:
            Tuple of (latent_representation, reconstruction)
        """
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstruction = self.decoder(latent)
        
        return latent, reconstruction
    
    def encode(self, x):
        """
        Encode input to latent space
        """
        return self.encoder(x)
    
    def decode(self, latent):
        """
        Decode from latent space
        """
        return self.decoder(latent)


if __name__ == "__main__":
    # Test the autoencoder
    batch_size = 8
    input_dim = 256
    
    model = Autoencoder(input_dim=input_dim)
    x = torch.randn(batch_size, input_dim)
    
    latent, reconstruction = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
