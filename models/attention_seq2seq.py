import torch
import torch.nn as nn

class AttentionSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=3, n_heads=4):
        super(AttentionSeq2Seq, self).__init__()
        # Output_dim=3 for 10th, 50th, and 90th quantiles
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        attn_out, _ = self.attention(enc_out, enc_out, enc_out)
        last_step = attn_out[:, -1, :] 
        quantiles = self.fc(last_step) # Returns [batch, 3]
        return quantiles, None