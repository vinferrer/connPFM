"""Functions to perform event detection."""
import logging
import subprocess
from os.path import basename, join

import numpy as np
from debiasing.debiasing_functions import debiasing_spike  # or debiasing_block
from joblib import Parallel, delayed
from nilearn.input_data import NiftiLabelsMasker
from connectivity.plotting import plot_ets_matrix
from scipy.stats import zscore
from utils import atlas_mod
from utils.hrf_generator import HRFMatrix

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

    return (rssr, np.min(etsr), np.max(etsr))


def event_detection(DATA_file, atlas, surrprefix="", sursufix="", segments=True):
    """Perform event detection on given data."""
    masker = NiftiLabelsMasker(
        labels_img=atlas,
        standardize=False,
        memory="nilearn_cache",
        strategy="mean",
    )

    data = masker.fit_transform(DATA_file)
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

    # repeat with randomized time series
    numrand = 100
    # initialize array for null rss
    rssr = np.zeros([t, numrand])

    results = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(rss_surr)(z_ts, u, v, surrprefix, sursufix, masker, irand)
        for irand in range(numrand)
    )

    for irand in range(numrand):
        rssr[:, irand] = results[irand][0]

    # TODO: find out why there is such a big peak on time-point 0 for AUC surrogates
    if "AUC" in surrprefix:
        rssr[0, :] = 0
        hist_ranges = np.zeros((2, numrand))
        for irand in range(numrand):
            hist_ranges[0, irand] = results[irand][1]
            hist_ranges[1, irand] = results[irand][2]

        hist_min = np.min(hist_ranges, axis=1)[0]
        hist_max = np.max(hist_ranges, axis=1)[1]

    p = np.zeros([t, 1])
    rssr_flat = rssr.flatten()
    for i in range(t):
        p[i] = np.mean(rssr_flat >= rss[i])
    # apply statistical cutoff
    pcrit = 0.001

    # find frames that pass statistical testz_ts
    idx = np.argwhere(p < pcrit)[:, 0]
    if segments:
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
            numrand=numrand,
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


def debiasing(data_file, mask, mtx, tr, out_dir, history_str):
    """Perform debiasing based on denoised edge-time matrix."""
    LGR.info("Performing debiasing based on denoised edge-time matrix...")
    masker = NiftiLabelsMasker(
        labels_img=mask,
        standardize=False,
        memory="nilearn_cache",
        strategy="mean",
    )

    # Read data
    data = masker.fit_transform(data_file)

    z_ts = np.nan_to_num(zscore(data, ddof=1))
    # Get number of time points/nodes
    [t, n] = z_ts.shape

    # calculate ets
    ets, idx_u, idx_v = calculate_ets(z_ts, n)
    # Generate mask of significant edge-time connections
    ets_mask = np.zeros(data.shape)
    idxs = np.where(mtx != 0)
    time_idxs = idxs[0]
    edge_idxs = idxs[1]

    LGR.info("Generating mask of significant edge-time connections...")
    for idx, time_idx in enumerate(time_idxs):
        ets_mask[time_idx, idx_u[edge_idxs[idx]]] = 1
        ets_mask[time_idx, idx_v[edge_idxs[idx]]] = 1

    # Create HRF matrix
    hrf = HRFMatrix(
        TR=tr,
        TE=[0],
        nscans=data.shape[0],
        r2only=True,
        is_afni=True,
    )
    hrf.generate_hrf()

    # Perform debiasing
    deb_output = debiasing_spike(hrf, data, ets_mask)
    beta = deb_output["beta"]
    fitt = deb_output["betafitts"]

    # Transform results back to 4D
    beta_4D = masker.inverse_transform(beta)
    beta_file = join(out_dir, f"{basename(data_file[:-7])}_beta_ETS.nii.gz")
    beta_4D.to_filename(beta_file)
    atlas_mod.inverse_transform(beta_file, data_file)
    subprocess.run(f"3dNotes {join(out_dir, beta_file)} -h {history_str}", shell=True)

    fitt_4D = masker.inverse_transform(fitt)
    fitt_file = join(out_dir, f"{basename(data_file[:-7])}_fitt_ETS.nii.gz")
    fitt_4D.to_filename(fitt_file)
    subprocess.run(f"3dNotes {join(out_dir, fitt_file)} -h {history_str}", shell=True)
    atlas_mod.inverse_transform(fitt_file, data_file)

    LGR.info("Debiasing finished and files saved.")

    return beta, fitt


def ev_workflow(DATAFILE,AUCFILE,ATLAS,SURR_DIR,OUT_DIR,DVARS=None,ENORM=None,afni_text=None,history_str=''):
    """
    Main function to perform event detection and plot results.
    """
    # Paths to files
    # Perform event detection on ORIGINAL data
    LGR.info("Performing event-detection on original data...")
    (
        ets_orig_sur,
        rss_orig_sur,
        rssr_orig_sur,
        idxpeak_orig_sur,
        etspeaks_orig_sur,
        mu_orig_sur,
        _,
        _,
        _,
    ) = event_detection(DATAFILE, ATLAS, join(SURR_DIR, "surrogate_"))

    # Perform event detection on AUC
    LGR.info("Performing event-detection on AUC...")
    (
        ets_auc,
        rss_auc,
        rssr_auc,
        idxpeak_auc,
        etspeaks_AUC,
        mu_AUC,
        ets_auc_denoised,
        idx_u,
        idx_v,
    ) = event_detection(AUCFILE, ATLAS, join(SURR_DIR, "surrogate_AUC_"))

    LGR.info("Plotting original, AUC, and AUC-denoised ETS matrices...")
    plot_ets_matrix(ets_orig_sur, OUT_DIR, "_original", DVARS, ENORM, idxpeak_auc)

    # Plot ETS and denoised ETS matrices of AUC
    plot_ets_matrix(ets_auc, OUT_DIR, "_AUC_original", DVARS, ENORM, idxpeak_auc)
    plot_ets_matrix(ets_auc_denoised, OUT_DIR, "_AUC_denoised", DVARS, ENORM, idxpeak_auc)

    # Save RSS time-series as text file for easier visualization on AFNI
    if afni_text != None:
        rss_out = np.zeros(rss_auc.shape)
        rss_out[idxpeak_auc] = rss_auc[idxpeak_auc]
        np.savetxt(afni_text, rss_out)
    np.savetxt(join(OUT_DIR, "ets_AUC_denoised.txt"),ets_auc_denoised)

