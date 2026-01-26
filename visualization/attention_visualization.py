"""
Attention weight visualization.
"""

import matplotlib.pyplot as plt

def plot_attention(attention_weights):
    weights = attention_weights.mean(0).detach().cpu().numpy()
    plt.imshow(weights, aspect="auto")
    plt.colorbar()
    plt.title("Attention Weights")
    plt.xlabel("Time Steps")
    plt.ylabel("Heads")
    plt.show()
