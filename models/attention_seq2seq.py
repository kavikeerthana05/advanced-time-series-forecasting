"""
Attention-based Seq2Seq model using multi-head self-attention.
"""

import torch
import torch.nn as nn

class AttentionSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, n_heads=4):
        super(AttentionSeq2Seq, self).__init__()
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        # Decoder/Output
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        enc_out, _ = self.encoder(x)
        
        # Self-attention: query, key, and value are all encoder outputs
        attn_out, attn_weights = self.attention(enc_out, enc_out, enc_out)
        
        # Global average pooling over the sequence or taking last step
        last_step = attn_out[:, -1, :] 
        out = self.fc(last_step)
        return out, attn_weights