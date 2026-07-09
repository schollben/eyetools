import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def create_subplot_grid(n_sesh):

    n_cols = min(int(np.ceil(np.sqrt(n_sesh))), 4)
    n_rows = int(np.ceil(n_sesh / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2*n_cols, 2*n_rows),
                             sharex=False,
                             squeeze=False)
    axes = axes.flatten()
    for ax in axes[n_sesh:]:
        ax.set_visible(False)
    sns.despine(fig=fig)
    fig.tight_layout(pad=2, h_pad=2, w_pad=2)
    return fig, axes
