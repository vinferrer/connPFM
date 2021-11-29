"""Main workflow to perform event detection."""
import logging
from os.path import join

import numpy as np
from joblib import Parallel, delayed
from nilearn.input_data import NiftiLabelsMasker
from scipy.stats import zscore

from connPFM.connectivity import utils
from connPFM.connectivity.plotting import plot_ets_matrix

LGR = logging.getLogger(__name__)


def event_detection(
    data_file,
    atlas,
    surrprefix="",
    sursufix="",
    nsur=100,
    segments=True,
    peak_detection="rss",
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
    ets, u, v = utils.calculate_ets(z_ts, n)

    # Initialize thresholded edge time-series matrix with zeros
    etspeaks = np.zeros(ets.shape)

    # Initialize array of indices of detected peaks
    idxpeak = np.zeros(t)

    # Initialize RSS with zeros
    rss = np.zeros(t)

    # Initialize array for null RSS
    rssr = np.zeros([t, nsur])

    # calculate ets and rss of surrogate data
    surrogate_events = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(utils.rss_surr)(z_ts, u, v, surrprefix, sursufix, masker, irand)
        for irand in range(nsur)
    )

    hist_ranges = np.zeros((2, nsur))
    for irand in range(nsur):
        hist_ranges[0, irand] = surrogate_events[irand][1]
        hist_ranges[1, irand] = surrogate_events[irand][2]

    hist_min = np.min(hist_ranges, axis=1)[0]
    hist_max = np.max(hist_ranges, axis=1)[1]

    # Make selection of points with RSS
    if "rss" in peak_detection:
        LGR.info("Selecting points with RSS...")
        # calculate rss
        rss = np.sqrt(np.sum(np.square(ets), axis=1))

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
            idxpeak = utils.remove_neighboring_peaks(rss, idx)
        # get activity at peak
        else:
            idxpeak = idx

        # get co-fluctuation at peaks
        etspeaks = utils.threshold_ets_matrix(ets, thr=0, selected_idxs=idxpeak)

    # Make selection of points with edge time-series matrix
    elif "ets" in peak_detection:
        LGR.info("Selecting points with edge time-series matrix...")
        if peak_detection == "ets":
            LGR.info("Reading AUC of surrogates to perform the thresholding step...")
            thr = utils.surrogates_histogram(
                surrprefix,
                sursufix,
                masker,
                hist_range=(hist_min, hist_max),
                numrand=nsur,
            )
        elif peak_detection == "ets_time":
            # Initialize array for threshold
            thr = np.zeros(t)
            for time_idx in range(t):
                thr[time_idx] = utils.surrogates_histogram(
                    surrprefix,
                    sursufix,
                    masker,
                    hist_range=(hist_min, hist_max),
                    numrand=nsur,
                    time_point=time_idx,
                )
        # Apply threshold on edge time-series matrix
        etspeaks = utils.threshold_ets_matrix(ets, thr)

    # calculate mean co-fluctuation (edge time series) across all peaks
    mu = np.nanmean(etspeaks, 0)

    return ets, rss, rssr, idxpeak, etspeaks, mu, u, v


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
    peak_detection="rss",
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
        ets_auc_denoised,
        _,
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
