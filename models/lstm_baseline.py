"""
Baseline LSTM Seq2Seq model.
"""

import torch
import torch.nn as nn

class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(LSTMBaseline, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        # Use last layer hidden state
        out = self.fc(hn[-1])
        return out, None # Return None for attention weights consistency