"""
Baseline LSTM Seq2Seq model.
"""

import torch.nn as nn

class LSTMSeq2Seq(nn.Module):
    """
    Baseline LSTM model for multivariate time series forecasting.
    """

    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, horizon):
        _, (h, c) = self.encoder(x)
        decoder_input = x[:, -1:, :]
        outputs = []

        for _ in range(horizon):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            pred = self.fc(out)
            outputs.append(pred)
            decoder_input = pred

        return nn.functional.concat(outputs, dim=1)
