"""Ve bieu do ket qua."""

import matplotlib.pyplot as plt


def plot_prediction_scatter(y_true, y_pred):
    """Ve bieu do so sanh gia tri thuc va du doan."""
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.7)
    ax.set_xlabel("Gia tri thuc")
    ax.set_ylabel("Gia tri du doan")
    ax.set_title("So sanh gia tri thuc va du doan")
    return fig, ax
