"""Plotting script for event detection."""
import logging
from os.path import join as opj

import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
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


def plot_comparison(
    rss_orig_sur,
    rssr_orig_sur,
    idxpeak_orig_sur,
    rss_fitt,
    rssr_fitt,
    idxpeak_fitt,
    rss_beta,
    rssr_beta,
    idxpeak_beta,
    rss_auc,
    rssr_auc,
    idxpeak_auc,
    ats,
    outdir,
):
    """
    Plot comparison of different RSS with vertical subplots.
    """
    greymap = cm.get_cmap("Greys")
    colors = greymap(np.linspace(0, 0.65, rssr_orig_sur.shape[1]))

    min_range = np.min(np.minimum(rss_orig_sur, rss_fitt)) * 0.9
    max_range = np.max(np.maximum(rss_orig_sur, rss_fitt)) * 1.1

    _, axs = plt.subplots(5, 1, figsize=FIGSIZE)
    for i in range(rssr_orig_sur.shape[1]):
        axs[0].plot(rssr_orig_sur[:, i], color=colors[i], linewidth=0.5)
    axs[0].plot(
        idxpeak_orig_sur,
        rss_orig_sur[idxpeak_orig_sur],
        "r*",
        label="orig_sur-peaks",
        markersize=20,
    )
    axs[0].plot(
        rss_orig_sur,
        color="k",
        linewidth=3,
        label="orig_sur",
    )
    axs[0].set_ylim([min_range, max_range])
    axs[0].set_title("Original signal")

    for i in range(rssr_orig_sur.shape[1]):
        axs[1].plot(rssr_fitt[:, i], color=colors[i], linewidth=0.5)
    axs[1].plot(idxpeak_fitt, rss_fitt[idxpeak_fitt], "r*", label="fitt-peaks", markersize=20)
    axs[1].plot(rss_fitt, color="k", linewidth=3, label="fitt")
    axs[1].set_ylim([min_range, max_range])
    axs[1].set_title("Fitted signal")

    for i in range(rssr_orig_sur.shape[1]):
        axs[2].plot(rssr_beta[:, i], color=colors[i], linewidth=0.5)
    axs[2].plot(idxpeak_beta, rss_beta[idxpeak_beta], "r*", label="beta-peaks", markersize=20)
    axs[2].plot(rss_beta, color="k", linewidth=3, label="beta")
    axs[2].set_title("Betas")

    for i in range(rssr_orig_sur.shape[1]):
        axs[3].plot(rssr_auc[:, i], color=colors[i], linewidth=0.5)
    axs[3].plot(idxpeak_auc, rss_auc[idxpeak_auc], "r*", label="AUC-peaks", markersize=20)
    axs[3].plot(rss_auc, color="k", linewidth=3, label="AUC")
    axs[3].set_title("AUCs")

    axs[4].plot(ats, label="ATS", color="black")
    axs[4].set_title("Activation time-series")

    plt.legend()
    plt.savefig(opj(outdir, "event_detection.png"), dpi=300)


def plot_all(
    rss_orig_sur,
    idxpeak_orig_sur,
    rss_beta,
    idxpeak_beta,
    rss_fitt,
    idxpeak_fitt,
    outdir,
):
    """
    Plot all RSS lines on same figure.
    """
    plt.figure(figsize=FIGSIZE)

    # Original signal
    rss_orig_norm = (rss_orig_sur - rss_orig_sur.min()) / (rss_orig_sur.max() - rss_orig_sur.min())
    plt.plot(
        idxpeak_orig_sur,
        rss_orig_norm[idxpeak_orig_sur],
        "r*",
        "linewidth",
        3,
        label="orig_sur-peaks",
    )
    plt.plot(
        range(rss_orig_norm.shape[0]),
        rss_orig_norm,
        "k",
        "linewidth",
        3,
        label="orig_sur",
    )

    # Betas
    rss_beta_norm = (rss_beta - rss_beta.min()) / (rss_beta.max() - rss_beta.min())
    plt.plot(
        idxpeak_beta,
        rss_beta_norm[idxpeak_beta],
        "g*",
        "linewidth",
        3,
        label="deconvolved_peaks",
    )
    plt.plot(
        range(rss_beta_norm.shape[0]),
        rss_beta_norm,
        "b",
        "linewidth",
        3,
        label="deconvolved",
    )

    # Fitted signal
    rss_fitt_norm = (rss_fitt - rss_fitt.min()) / (rss_fitt.max() - rss_fitt.min())
    plt.plot(
        idxpeak_fitt,
        rss_fitt_norm[idxpeak_fitt],
        "m*",
        "linewidth",
        3,
        label="fitted_peaks",
    )
    plt.plot(
        range(rss_fitt_norm.shape[0]),
        rss_fitt_norm,
        "y",
        "linewidth",
        3,
        label="fitted",
    )
    plt.legend()
    plt.savefig(opj(outdir, "event_detection_all.png"), dpi=300)


def plot_ets_matrix(
    ets, outdir, sufix="", dvars_file=None, enorm_file=None, peaks=None, vmin=-0.5, vmax=0.5
):
    """
    Plots edge-time matrix
    """
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
        im = ax0.imshow(ets.T, vmin=vmin, vmax=vmax, cmap="bwr", aspect="auto")
        ax0.set_ylabel("Edge-edge connections")
        plt.colorbar(im, orientation="vertical", ax=ax0, cax=cax)  # ax=axs.ravel().tolist()
        dvars[1] = np.mean(dvars)
        ax1.plot(dvars)
        ax1.set_title("DVARS")
        ax1.margins(0, 0)
        for i in peaks:
            ax1.axvspan(i, i + 1, facecolor="b", alpha=0.5)
            ax2.axvspan(i, i + 1, facecolor="b", alpha=0.5)
        ax2.plot(enorm)
        ax2.set_title("ENORM")
        ax2.set_xlabel("Time (TR)")
        ax2.margins(0, 0)
        plt.savefig(opj(outdir, f"ets{sufix}.png"), dpi=300)
    else:
        fig, axs = plt.subplots(1, 1, figsize=FIGSIZE)
        plt.imshow(ets.T, vmin=vmin, vmax=vmax, cmap="bwr", aspect="auto")
        plt.title("Edge-time series")
        plt.xlabel("Time (TR)")
        plt.ylabel("Edge-edge connections")
        plt.colorbar()
        plt.savefig(opj(outdir, f"ets{sufix}.png"), dpi=300)
