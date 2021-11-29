"""Functions to perform event detection."""
import logging
from os.path import join

import numpy as np
from joblib import Parallel, delayed
from nilearn.input_data import NiftiLabelsMasker
from scipy.stats import zscore

from connPFM.connectivity.plotting import plot_ets_matrix

LGR = logging.getLogger(__name__)


def calculate_ets(y, n):
    """Calculate edge-time series."""
    # upper triangle indices (node pairs = edges)
    u, v = np.argwhere(np.triu(np.ones(n), 1)).T

    # edge time series
    ets = y[:, u] * y[:, v]

    return ets, u, v


def rss_surr(z_ts, u, v, surrprefix, sursufix, masker, irand):
    """Calculate RSS on surrogate data."""
    [t, n] = z_ts.shape

    if surrprefix != "":
        zr = masker.fit_transform(f"{surrprefix}{irand}{sursufix}.nii.gz")
        if "AUC" not in surrprefix:
            zr = np.nan_to_num(zscore(zr, ddof=1))

        # TODO: find out why surrogates of AUC have NaNs after reading data with masker.
        zr = np.nan_to_num(zr)
    else:
        # perform numrand randomizations
        zr = np.copy(z_ts)
        for i in range(n):
            zr[:, i] = np.roll(zr[:, i], np.random.randint(t))

    # edge time series with circshift data
    etsr = zr[:, u] * zr[:, v]

    # calcuate rss
    rssr = np.sqrt(np.sum(np.square(etsr), axis=1))

    return (rssr, etsr, np.min(etsr), np.max(etsr))


def remove_neighboring_peaks(rss, idx):
    """
    Identify contiguous peaks among selected points in the RSS vector.

    Parameters
    ----------
    rss : ndarray
        RSS vector.
    idx : ndarray
        Indices of the selected peaks.

    Returns
    -------
    idxpeak: ndarray
        Indices of the selected peaks with no neighboring points.
    """
    # identify contiguous segments of frames that pass statistical test
    dff = idx.T - range(len(idx))
    unq = np.unique(dff)
    nevents = len(unq)

    # find the peak rss within each segment
    idxpeak = np.zeros([nevents, 1])
    for ievent in range(nevents):
        idxevent = idx[dff.T == unq[ievent].T]
        rssevent = rss[idxevent]
        idxmax = np.argmax(rssevent)
        idxpeak[ievent] = idxevent[idxmax]
    idxpeak = idxpeak[:, 0].astype(int)
    return idxpeak


def threshold_ets_matrix(ets_matrix, thr, selected_idxs=None):
    """
    Threshold the edge time-series matrix based on the selected time-points and
    the surrogate matrices.
    """
    # Initialize matrix with zeros
    thresholded_matrix = np.zeros(ets_matrix.shape)

    # Get selected columns from ETS matrix
    if selected_idxs is not None:
        thresholded_matrix[selected_idxs, :] = ets_matrix[selected_idxs, :]
    else:
        thresholded_matrix = ets_matrix

    # Threshold ETS matrix based on surrogate percentile
    thresholded_matrix[abs(thresholded_matrix) < thr] = 0

    return thresholded_matrix


def calculate_hist(surrprefix, sursufix, irand, masker, hist_range, nbins=500):
    """Calculate histogram."""
    auc = masker.fit_transform(f"{surrprefix}{irand}{sursufix}.nii.gz")
    [t, n] = auc.shape
    ets_temp, _, _ = calculate_ets(np.nan_to_num(auc), n)

    ets_hist, bin_edges = np.histogram(ets_temp.flatten(), bins=nbins, range=hist_range)

    return (ets_hist, bin_edges)


def surrogates_to_array(
    surrprefix, sursufix, masker, hist_range, numrand=100, nbins=500, percentile=95
):
    """
    Read AUCs of surrogates, calculate histogram and sum of all histograms to
    obtain a single histogram that summarizes the data.
    """
    ets_hist = np.zeros((numrand, nbins))

    hist = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(calculate_hist)(surrprefix, sursufix, irand, masker, hist_range, nbins)
        for irand in range(numrand)
    )

    for irand in range(numrand):
        ets_hist[irand, :] = hist[irand][0]

    bin_edges = hist[0][1]

    ets_hist_sum = np.sum(ets_hist, axis=0)
    cumsum_percentile = np.cumsum(ets_hist_sum) / np.sum(ets_hist_sum) * 100
    thr = bin_edges[len(cumsum_percentile[cumsum_percentile <= percentile])]

    return thr


def event_detection(
    data_file, atlas, peak_detection="rss", surrprefix="", sursufix="", nsur=100, segments=True
):
    """Perform event detection on given data."""
    masker = NiftiLabelsMasker(
        labels_img=atlas,
        standardize=False,
        strategy="mean",
    )

    data = masker.fit_transform(data_file)
    # load and zscore time series
    # AUC does not get z-scored
    if "AUC" in surrprefix:
        z_ts = data
    else:
        z_ts = np.nan_to_num(zscore(data, ddof=1))
    # Get number of time points/nodes
    [t, n] = z_ts.shape

    # calculate ets
    ets, u, v = calculate_ets(z_ts, n)

    # calculate ets and rss of surrogate data
    surrogate_events = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(rss_surr)(z_ts, u, v, surrprefix, sursufix, masker, irand) for irand in range(nsur)
    )

    hist_ranges = np.zeros((2, nsur))
    for irand in range(nsur):
        hist_ranges[0, irand] = surrogate_events[irand][2]
        hist_ranges[1, irand] = surrogate_events[irand][3]

    hist_min = np.min(hist_ranges, axis=1)[0]
    hist_max = np.max(hist_ranges, axis=1)[1]

    # Make selection of points with RSS
    if "rss" in peak_detection:
        # calculate rss
        rss = np.sqrt(np.sum(np.square(ets), axis=1))
        # initialize array for null rss
        rssr = np.zeros([t, nsur])

        for irand in range(nsur):
            rssr[:, irand] = surrogate_events[irand][0]

        # TODO: find out why there is such a big peak on time-point 0 for AUC surrogates
        if "AUC" in surrprefix:
            rssr[0, :] = 0

        # Initialize p-values array
        p = np.zeros([t, 1])

        # Statistical cutoff
        pcrit = 0.001

        # Find peaks with a general threshold for RSS
        if peak_detection == "rss":
            rssr_flat = rssr.flatten()
            for i in range(t):
                p[i] = np.mean(rssr_flat >= rss[i])

        # Find peaks with a different threshold for each time-point in RSS
        elif peak_detection == "rss_time":
            for i in range(t):
                p[i] = np.mean(np.squeeze(rssr[i, :]) >= rss[i])

        # find frames that pass statistical testz_ts
        idx = np.argwhere(p < pcrit)[:, 0]
        if segments:
            idxpeak = remove_neighboring_peaks(rss, idx)
        # get activity at peak
        else:
            idxpeak = idx

    # Make selection of points with edge time-series matrix
    elif "ets" in peak_detection:
        if peak_detection == "ets":
            idxpeak = 1
        elif peak_detection == "ets_time":
            idxpeak = 1

    tspeaks = z_ts[idxpeak, :]

    # get co-fluctuation at peak
    etspeaks = tspeaks[:, u] * tspeaks[:, v]
    # calculate mean co-fluctuation (edge time series) across all peaks
    mu = np.nanmean(etspeaks, 0)

    if "AUC" in surrprefix:
        LGR.info("Reading AUC of surrogates to perform the thresholding step...")
        ets_thr = surrogates_to_array(
            surrprefix,
            sursufix,
            masker,
            hist_range=(hist_min, hist_max),
            numrand=nsur,
        )
        ets_thr = threshold_ets_matrix(ets, idxpeak, ets_thr)
    else:
        ets_thr = None

    return ets, rss, rssr, idxpeak, etspeaks, mu, ets_thr, u, v


def ev_workflow(
    data_file,
    auc_file,
    atlas,
    surr_dir,
    out_dir,
    nsurrogates=100,
    dvars=None,
    enorm=None,
    afni_text=None,
    history_str="",
):
    """
    Main function to perform event detection and plot results.
    """
    # Paths to files
    # Perform event detection on ORIGINAL data
    LGR.info("Performing event-detection on original data...")
    ets_orig_sur = event_detection(
        data_file, atlas, join(surr_dir, "surrogate_"), nsur=nsurrogates
    )[0]

    # Perform event detection on AUC
    LGR.info("Performing event-detection on AUC...")
    (
        ets_auc,
        rss_auc,
        _,
        idxpeak_auc,
        _,
        _,
        ets_auc_denoised,
        _,
        _,
    ) = event_detection(auc_file, atlas, join(surr_dir, "surrogate_AUC_"), nsur=nsurrogates)

    LGR.info("Plotting original, AUC, and AUC-denoised ETS matrices...")
    plot_ets_matrix(ets_orig_sur, out_dir, "_original", dvars, enorm, idxpeak_auc)

    # Plot ETS and denoised ETS matrices of AUC
    plot_ets_matrix(ets_auc, out_dir, "_AUC_original", dvars, enorm, idxpeak_auc)
    plot_ets_matrix(ets_auc_denoised, out_dir, "_AUC_denoised", dvars, enorm, idxpeak_auc)

    # Save RSS time-series as text file for easier visualization on AFNI
    if afni_text is not None:
        rss_out = np.zeros(rss_auc.shape)
        rss_out[idxpeak_auc] = rss_auc[idxpeak_auc]
        np.savetxt(afni_text, rss_out)
    np.savetxt(join(out_dir, "ets_AUC_denoised.txt"), ets_auc_denoised)
    return ets_auc_denoised
