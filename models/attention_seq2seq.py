"""
Attention-based Seq2Seq model using multi-head self-attention.
"""

import torch
import torch.nn as nn

class AttentionSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, n_heads=4, num_layers=2):
        super(AttentionSeq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        # Encoder LSTM
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        
        # Layer Norm for stability
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        enc_out, _ = self.encoder(x)
        
        # Self-attention
        attn_out, attn_weights = self.attention(enc_out, enc_out, enc_out)
        attn_out = self.ln(attn_out + enc_out) # Residual connection
        
        # Pull the last time step for prediction
        last_step = attn_out[:, -1, :] 
        out = self.fc(last_step)
        return out
    
    def get_attention_weights(self, x):
        self.eval()
        with torch.no_grad():
            enc_out, _ = self.encoder(x)
            _, attn_weights = self.attention(enc_out, enc_out, enc_out)
        return attn_weights