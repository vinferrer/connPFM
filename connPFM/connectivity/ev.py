"""Functions to perform event detection."""
import logging
from os.path import join

import numpy as np
from joblib import Parallel, delayed
from nilearn.input_data import NiftiLabelsMasker
from scipy.stats import zscore

from connPFM.connectivity.plotting import plot_ets_matrix

LGR = logging.getLogger(__name__)


def calculate_p_value(original, surrogate):
    """Calculate p-value from surrogate data.

    Parameters
    ----------
    original : array
        original RSS
    surrogate : array
        surrogate RSS matrix

    Returns
    -------
    array
        p-value for each frame
    """

    # Flatten array if it isn't already
    if surrogate.shape[1] > 1:
        surrogate = surrogate.flatten()

    # Calculate p-value of original being higher than surrogate.
    p = np.zeros(original.shape[0])
    for i in range(original.shape[0]):
        p[i] = np.mean(surrogate > original[i])

    return p


def find_significant_frames(p, pcrit=0.001, segments=True, rss=None):
    """Find frames that are statistically significant.

    Parameters
    ----------
    p : array
        p-value at each frame
    pcrit : float, optional
        significance threshold, by default 0.001
    segments : bool, optional
        consider segments instead of sole points, by default True
    rss : array, optional
        array with the RSS, by default None

    Returns
    -------
    array
        indexes of the peak RSS values
    """

    idx = np.argwhere(p < pcrit)[:, 0]
    if segments and (rss is not None):
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
    # get activity at peak
    else:
        idxpeak = idx

    return idxpeak


def circular_shift_randomization(y, n, t):
    """Perform randomization of time series."""
    z = np.copy(y)
    for i in range(n):
        z[:, i] = np.roll(z[:, i], np.random.randint(t))

    return z


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
        # Perform circular shift randomization
        zr = circular_shift_randomization(z_ts, n, t)

    # edge time series with circshift data
    etsr = zr[:, u] * zr[:, v]

    # calcuate rss
    rssr = np.sqrt(np.sum(np.square(etsr), axis=1))

    return (rssr, np.min(etsr), np.max(etsr))


def event_detection(data_file, atlas, surrprefix="", sursufix="", nsur=100, segments=True):
    """Perform event detection on given data."""
    masker = NiftiLabelsMasker(
        labels_img=atlas,
        standardize=False,
        memory="nilearn_cache",
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

    # calculate rss
    rss = np.sqrt(np.sum(np.square(ets), axis=1))
    # initialize array for null rss
    rssr = np.zeros([t, nsur])

    results = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(rss_surr)(z_ts, u, v, surrprefix, sursufix, masker, irand) for irand in range(nsur)
    )

    for irand in range(nsur):
        rssr[:, irand] = results[irand][0]

    # TODO: find out why there is such a big peak on time-point 0 for AUC surrogates
    if "AUC" in surrprefix:
        rssr[0, :] = 0
        hist_ranges = np.zeros((2, nsur))
        for irand in range(nsur):
            hist_ranges[0, irand] = results[irand][1]
            hist_ranges[1, irand] = results[irand][2]

        hist_min = np.min(hist_ranges, axis=1)[0]
        hist_max = np.max(hist_ranges, axis=1)[1]

    # Calculate p-values and find significant frames
    p = calculate_p_value(rss, rssr)
    idxpeak = find_significant_frames(p, segments=segments)

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


def threshold_ets_matrix(ets_matrix, selected_idxs, thr):
    """
    Threshold the edge time-series matrix based on the selected time-points and
    the surrogate matrices.
    """
    # Initialize matrix with zeros
    thresholded_matrix = np.zeros(ets_matrix.shape)

    # Get selected columns from ETS matrix
    thresholded_matrix[selected_idxs, :] = ets_matrix[selected_idxs, :]

    # Threshold ETS matrix based on surrogate percentile
    thresholded_matrix[thresholded_matrix < thr] = 0

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
