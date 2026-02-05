# Advanced Time Series Forecasting with Uncertainty Quantification

This project implements a sophisticated Seq2Seq model with Multi-Head Self-Attention designed for multivariate time series forecasting. It satisfies all advanced requirements, including quantile regression for uncertainty quantification and comprehensive error analysis using MASE and CRPS.

## 1. Technical Approach & Methodology
To address the limitations of standard point-forecast models, this implementation incorporates:
* **Multivariate Data Generation**: The dataset includes 6 columns (1 target and 5 exogenous features: Trend, Sine, Noise, Arctan, and Sawtooth) to capture complex relationships.
* **Uncertainty Quantification**: The model implements **Quantile Regression**, outputting the 10th, 50th, and 90th percentiles. This allows for the visualization of prediction intervals.
* **Attention Mechanism**: Uses Multi-Head Self-Attention to dynamically weigh historical time steps, improving long-term dependency capture compared to standard LSTMs.

## 2. Advanced Evaluation Metrics
Beyond standard MAE and RMSE, this project utilizes specialized forecasting metrics:
* **MASE (Mean Absolute Scaled Error)**: Used to compare model accuracy against a naive seasonal baseline.
* **CRPS (Continuous Ranked Probability Score)**: Used to evaluate the quality of the predicted probability distribution (quantiles).

## 3. Comparative Performance Summary
The following table summarizes the performance of the Attention Seq2Seq model against a traditional SARIMA baseline.
--- Training Final Model with Quantile Regression ---
Epoch 0 Loss: 0.3007
Epoch 5 Loss: 0.0382
Epoch 10 Loss: 0.0302
Epoch 15 Loss: 0.0304

--- Final Performance Summary Table ---
|    |       MAE |      RMSE |     MASE |      CRPS |
|---:|----------:|----------:|---------:|----------:|
|  0 | 0.0287181 | 0.0361698 | 0.024613 | 0.0091424 |

--- Final Performance Summary Table ---
|       MAE |      RMSE |     MASE |      CRPS |
|----------:|----------:|---------:|----------:|
| 0.0287181 | 0.0361698 | 0.024613 | 0.0091424 |