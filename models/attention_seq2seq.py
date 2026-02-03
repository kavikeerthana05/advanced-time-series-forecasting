"""
Attention-based Seq2Seq model using multi-head self-attention.
"""

import torch.nn as nn

class AttentionSeq2Seq(nn.Module):
    """
    Sequence-to-sequence LSTM model with attention mechanism
    for long-term multivariate time series forecasting.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super().__init__()

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, horizon):
        enc_out, _ = self.encoder(x)
        attn_out, attn_weights = self.attention(enc_out, enc_out, enc_out)

        decoder_input = attn_out[:, -1:, :]
        h = None
        outputs = []

        for _ in range(horizon):
            out, h = self.decoder(decoder_input, h)
            pred = self.fc(out)
            outputs.append(pred)
            decoder_input = out

        return nn.functional.concat(outputs, dim=1), attn_weights
