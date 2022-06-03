"""Utility functions to perform event detection."""
import logging

import numpy as np
from scipy.sparse import csr_matrix

LGR = logging.getLogger(__name__)


def sparse_histogram(sparse_matrix, bins, range):
    """Calculate histogram of sparse matrix."""
    hist, bin_edges = np.histogram(sparse_matrix.data, bins=bins, range=range)
    # Correct actual value of zero bin values
    zeros_in_data = np.sum(sparse_matrix.data==0)
    total_e = sparse_matrix.shape[0] * sparse_matrix.shape[1]
    zero_e = total_e - sparse_matrix.count_nonzero() - zeros_in_data
    hist[0] = zero_e + hist[0]

    return hist, bin_edges


def calculate_ets(y, n):
    """Calculate edge-time series."""
    # upper triangle indices (node pairs = edges)
    u, v = np.argwhere(np.triu(np.ones(n), 1)).T
    y_u = csr_matrix(y[:, u])
    y_v = csr_matrix(y[:, v])
    # edge time series
    ets = y_u.multiply(y_v)

    return ets, u, v


def rss_surr(z_ts, u, v, surrprefix, sursufix, masker, irand, nbins, hist_range=(0, 1)):
    """Calculate RSS on surrogate data."""
    [t, n] = z_ts.shape

    if surrprefix != "":
        zr = masker.fit_transform(f"{surrprefix}{irand}{sursufix}.nii.gz")

        # TODO: find out why surrogates of AUC have NaNs after reading data with masker.
        zr = np.nan_to_num(zr)
    else:
        # perform numrand randomizations
        zr = np.copy(z_ts)
        for i in range(n):
            zr[:, i] = np.roll(zr[:, i], np.random.randint(t))

    # edge time series with circshift data
    zr_u = csr_matrix(zr[:, u])
    zr_v = csr_matrix(zr[:, v])
    etsr = zr_u.multiply(zr_v)

    # calcuate rss
    rssr = np.array(np.sqrt(etsr.power(2).sum(axis=1)[:, 0].flatten())).flatten()

    # Calculate histogram
    ets_hist, bin_edges = sparse_histogram(etsr, nbins, hist_range)

    return (rssr, etsr, ets_hist, bin_edges)


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
    thresholded_matrix = csr_matrix(np.zeros(ets_matrix.shape))
    # Get selected columns from ETS matrix
    if selected_idxs is not None:
        thresholded_matrix[selected_idxs, :] = ets_matrix[selected_idxs, :]
    else:
        thresholded_matrix = ets_matrix

    # Threshold ETS matrix based on surrogate percentile
    # if thr is not an array, subtract it from the matrix
    if type(thr) is not np.ndarray:
        thr_mtx = csr_matrix(np.ones(thresholded_matrix.shape) * thr)
        thresholded_matrix = thresholded_matrix - thr_mtx
    else:
        thresholded_matrix -= thr[:, None]
        thresholded_matrix = csr_matrix(thresholded_matrix)
    thresholded_matrix[thresholded_matrix < 0] = 0
    # reconvert so compression is done properly
    thresholded_matrix = csr_matrix(thresholded_matrix.toarray())
    return thresholded_matrix


def calculate_surrogate_ets(surrprefix, sursufix, irand, masker):
    """Read surrogate data."""
    auc = masker.fit_transform(f"{surrprefix}{irand}{sursufix}.nii.gz")
    [t, n] = auc.shape
    ets, _, _ = calculate_ets(np.nan_to_num(auc), n)

    return ets


def calculate_hist(
    surrprefix,
    sursufix,
    irand,
    masker,
    hist_range,
    nbins=500,
):
    """Calculate histogram."""
    ets_temp = calculate_surrogate_ets(surrprefix, sursufix, irand, masker)
    # data works properly for the histogram except for the zero values
    ets_hist, bin_edges = sparse_histogram(ets_temp, nbins, hist_range)

    return (ets_hist, bin_edges)


def calculate_hist_threshold(hist, bins, percentile=95):
    """Calculate histogram threshold."""
    cumsum_percentile = np.cumsum(hist) / np.sum(hist) * 100
    thr = bins[len(cumsum_percentile[cumsum_percentile <= percentile])]

    return thr


def sum_histograms(
    hist_list,
):
    """
    Get histograms of all surrogates and sum them to
    obtain a single histogram that summarizes the data.
    """

    # Initialize matrix to store surrogate histograms
    all_hists = np.zeros((len(hist_list), hist_list[0][3].shape[0] - 1))

    for rand_idx in range(len(hist_list)):
        all_hists[rand_idx, :] = hist_list[rand_idx][2]

    ets_hist_sum = np.sum(all_hists, axis=0)

    return ets_hist_sum
