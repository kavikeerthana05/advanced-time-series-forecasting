"""
Attention weight visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_map(attn_weights, sample_idx=0):
    weights = attn_weights[sample_idx].detach().cpu().numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(weights, cmap='hot')
    plt.title("Attention Weight Map")
    plt.xlabel("Time Step (Key)")
    plt.ylabel("Time Step (Query)")
    plt.savefig('evaluation/attention_map.png')
    plt.close()

