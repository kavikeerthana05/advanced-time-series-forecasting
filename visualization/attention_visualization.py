"""
Attention weight visualization.
"""

import matplotlib.pyplot as plt
import torch


def plot_attention_weights(attention_weights: torch.Tensor):
    """
    Plot attention weights for a single sample.

    Parameters
    ----------
    attention_weights : torch.Tensor
        Shape: (target_steps, source_steps)
    """
    weights = attention_weights.detach().cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.imshow(weights, aspect="auto", cmap="viridis")
    plt.colorbar(label="Attention Weight")
    plt.xlabel("Source Time Steps")
    plt.ylabel("Target Time Steps")
    plt.title("Attention Weight Visualization")
    plt.tight_layout()
    plt.show()

