"""Plotting script for event detection."""
import logging
from os.path import join as opj

import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

fontsize = 28
params = {
    "legend.fontsize": fontsize,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
}
pylab.rcParams.update(params)
# Global variables
NRAND = 100
TR = 0.83
FIGSIZE = (45, 30)
HISTORY = "Deconvolution based on event-detection."
# Font size for plots
font = {"weight": "normal", "size": 28}
matplotlib.rc("font", **font)

LGR = logging.getLogger(__name__)


def plot_ets_matrix(
    ets, outdir, sufix="", dvars_file=None, enorm_file=None, peaks=None, vmin=None, vmax=None
):
    """
    Plots edge-time matrix
    """
    if vmin == None:
        vmin = np.min(ets)
    if vmax == None:
        vmax = np.max(ets) / 2
    if dvars_file is not None and enorm_file is not None:
        # Plot ETS matrix of original signal
        dvars = np.loadtxt(dvars_file)
        enorm = np.loadtxt(opj(enorm_file))
        # widths = [1]
        # heights = [2, 1, 1]
        # gs = dict(width_ratios=widths, height_ratios=heights)
        # fig, axs = plt.subplots(3, 1, figsize=FIGSIZE,gridspec_kw=gs)
        # im = axs[0].imshow(ets.T, vmin=vmin, vmax=vmax, cmap="bwr", aspect="auto")
        # axs[0].set_title("Edge-time series")
        # axs[0].set_ylabel("Edge-edge connections")
        # fig.colorbar(im, orientation="vertical", ax=axs[0]) # ax=axs.ravel().tolist()
        # axs[1].plot(dvars)
        # axs[1].set_title("DVARS")
        # axs[2].plot(enorm)
        # axs[2].set_title("ENORM")
        # axs[2].set_xlabel("Time (TR)")
        fig = plt.subplots(figsize=FIGSIZE)
        ax0 = plt.subplot(111)
        divider = make_axes_locatable(ax0)
        ax1 = divider.append_axes("bottom", size="25%", pad=1)
        ax2 = divider.append_axes("bottom", size="25%", pad=1)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        im = ax0.imshow(ets.T, vmin=vmin, vmax=vmax, cmap="OrRd", aspect="auto")
        ax0.set_ylabel("Edge-edge connections")
        plt.colorbar(im, orientation="vertical", ax=ax0, cax=cax)  # ax=axs.ravel().tolist()
        dvars[1] = np.mean(dvars)
        ax1.plot(dvars)
        ax1.set_title("DVARS")
        ax1.margins(0, 0)
        for i in peaks:
            ax1.axvspan(i, i + 1, facecolor="g", alpha=0.5)
            ax2.axvspan(i, i + 1, facecolor="g", alpha=0.5)
        ax2.plot(enorm)
        ax2.set_title("ENORM")
        ax2.set_xlabel("Time (TR)")
        ax2.margins(0, 0)
        plt.savefig(opj(outdir, f"ets{sufix}.png"), dpi=300)
    else:
        fig, _ = plt.subplots(1, 1, figsize=FIGSIZE)
        plt.imshow(ets.T, vmin=vmin, vmax=vmax, cmap="OrRd", aspect="auto")
        plt.title("Edge-time series")
        plt.xlabel("Time (TR)")
        plt.ylabel("Edge-edge connections")
        plt.colorbar()
        plt.savefig(opj(outdir, f"ets{sufix}.png"), dpi=300)
