# Advanced Time Series Forecasting with Attention

## Overview
This project implements and compares two deep learning models for multistep
time-series forecasting:

1. LSTM Baseline Model  
2. Attention-based Seq2Seq Model (LSTM + Self-Attention)

The goal is to evaluate how attention mechanisms improve long-term dependency
modeling compared to a standard LSTM baseline.

---
Note: Model interpretability is achieved via attention weight visualization,
which highlights influential timesteps during prediction. SHAP was not applied
as the task focuses on sequence-based interpretability rather than feature attribution.


## Project Structure

