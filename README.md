# Advanced Time Series Forecasting with Attention

This project compares a standard LSTM against a Seq2Seq model with Multi-Head Self-Attention.

## Steps to Run:
1. `pip install -r requirements.txt`
2. Run `python main.py`
3. Check `evaluation/` for metric outputs and attention maps.

## Model Choice:
The Attention model uses `nn.MultiheadAttention` to capture long-term dependencies across the 50-step window more effectively than the standard LSTM.