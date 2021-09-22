""" --- Empty description --- """

# Imports
# =======
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as tk
from matplotlib.colors import to_rgba


# ------------------------------------------------------------------
def plot_valves(VALVES, PARAM, figpath=None):
    """
    Plots valve system
    """
    X = np.linspace(0, 1, PARAM['Nx']+1)

    fig, ax = plt.subplots(figsize=np.array([15, 4])/2.54)

    # --> Create ax and fix ticks
    ax.set_xlim(-0.02, 1.02) ; ax.set_ylim(0, 1)
    ax.set_xticks([0, 1])
    ax.tick_params(left=False, labelleft=False)
    ax.set_xlabel('$x$ along domain (scaled)')

    ax.set_title(r'$N={:d}$ valves system'.format(len(VALVES['idx'])))

    # --> Plot distributions
    for xv in X[VALVES['idx']]:
        ax.axvspan(xv, xv+VALVES['width'][0], fc=to_rgba('k', 0.5), ec=to_rgba('k', 0))

    plt.tight_layout()

    if figpath is not None:
        print('Saving at {:}'.format(figpath))
        plt.savefig(figpath)

    plt.show()
    return fig, ax


# ------------------------------------------------------------------
def plot_flux_space(PARAM, VALVES, figpath=None):
    plt.tight_layout()

        if figpath is not None:
            print('Saving at {:}'.format(figpath))
            plt.savefig(figpath)

        plt.show()
        return fig, ax

