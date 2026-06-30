import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def create_subplot_grid(n_sesh):

    m = int(np.ceil(n_sesh / 2))
    fig, axes = plt.subplots(2, m, 
                             figsize=(2*m, 4), 
                             sharex=False, 
                             squeeze=False)
    axes = axes.flatten()
    sns.despine(fig=fig)
    fig.tight_layout(pad=2, h_pad=2, w_pad=2)
    return fig, axes
