"""
Transformer Encoder Module for Audio Feature Processing
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer to capture temporal information
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for processing audio features
    """
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=6, 
                 dim_feedforward=1024, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            src_mask: Attention mask
            src_key_padding_mask: Padding mask
            
        Returns:
            Encoded features of shape [batch_size, seq_len, d_model]
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Transpose for transformer: [seq_len, batch, d_model]
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(
            x, 
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Transpose back: [batch, seq_len, d_model]
        x = x.transpose(0, 1)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x


if __name__ == "__main__":
    # Test the transformer encoder
    batch_size = 8
    seq_len = 100
    input_dim = 128
    
    model = TransformerEncoder(input_dim=input_dim)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
