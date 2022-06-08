from os.path import join
from re import M

import numpy as np
from nilearn.input_data import NiftiLabelsMasker
from scipy.stats import zscore
from scipy.sparse import csr_matrix, save_npz

from connPFM.debiasing.debiasing import debiasing
from connPFM.debiasing.debiasing_functions import debiasing_block, debiasing_spike
from connPFM.utils.hrf_generator import HRFMatrix


def test_debiasing_spike(
    hrf_file, bold_file, atlas_file, ets_auc_denoised_file, beta_file, fitt_file, fitt_group_file
):
    masker = NiftiLabelsMasker(
        labels_img=atlas_file,
        standardize=False,
        strategy="mean",
    )
    # Read data
    data = masker.fit_transform(bold_file)

    z_ts = np.nan_to_num(zscore(data, ddof=1))
    # Get number of time points/nodes
    [t, n] = z_ts.shape

    # Get ETS indexes
    idx_u, idx_v = np.argwhere(np.triu(np.ones(n), 1)).T

    # Generate mask of significant edge-time connections
    ets_mask = np.zeros(data.shape)
    idxs = np.where(np.loadtxt(ets_auc_denoised_file) != 0)
    time_idxs = idxs[0]
    edge_idxs = idxs[1]

    for idx, time_idx in enumerate(time_idxs):
        ets_mask[time_idx, idx_u[edge_idxs[idx]]] = 1
        ets_mask[time_idx, idx_v[edge_idxs[idx]]] = 1

    # Create HRF matrix
    hrf = HRFMatrix(
        TR=1,
        TE=[0],
        nscans=data.shape[0],
        r2only=True,
        is_afni=True,
    )
    hrf.generate_hrf()
    deb_output = debiasing_spike(hrf, data, ets_mask)
    assert np.allclose(deb_output["beta"], masker.fit_transform(beta_file))
    assert np.allclose(deb_output["betafitts"], masker.fit_transform(fitt_file))
    deb_group = debiasing_spike(hrf, data, ets_mask, groups=True)
    assert np.allclose(deb_group["betafitts"], np.loadtxt(fitt_group_file))


def test_debiasing_block(
    hrf_file, AUC_file, bold_file, atlas_file, ets_auc_denoised_file, beta_block_file
):
    masker = NiftiLabelsMasker(
        labels_img=atlas_file,
        standardize=False,
        strategy="mean",
    )
    # Read data
    data = masker.fit_transform(bold_file)

    z_ts = np.nan_to_num(zscore(data, ddof=1))
    # Get number of time points/nodes
    [t, n] = z_ts.shape

    # Get ETS indexes
    idx_u, idx_v = np.argwhere(np.triu(np.ones(n), 1)).T

    # Generate mask of significant edge-time connections
    ets_mask = np.zeros(data.shape)
    idxs = np.where(np.loadtxt(ets_auc_denoised_file) != 0)
    time_idxs = idxs[0]
    edge_idxs = idxs[1]

    for idx, time_idx in enumerate(time_idxs):
        ets_mask[time_idx, idx_u[edge_idxs[idx]]] = 1
        ets_mask[time_idx, idx_v[edge_idxs[idx]]] = 1

    # Create HRF matrix
    hrf = HRFMatrix(TR=1, TE=[0], nscans=data.shape[0], r2only=True, is_afni=True, block=True)
    hrf.generate_hrf()
    (beta, S) = debiasing_block(masker.fit_transform(AUC_file), hrf.hrf, data, True)
    assert np.allclose(beta, np.loadtxt(beta_block_file))


def test_debias_ME(ME_files, ME_mask_2, ME_lars, ME_debias, testpath):
    auc = np.load(ME_lars[2])
    mtx = auc * (auc > np.percentile(auc, 75))
    mtx_file = join(testpath, "debias_ME_mtx.npz")
    save_npz(mtx_file, csr_matrix(mtx))
    beta, fitt = debiasing(
        ME_files[:-1],
        ME_mask_2,
        [15.4, 29.7, 44.0, 58.37, 2.6],
        mtx_file,
        2,
        join(testpath, "ME"),
        True,
        3,
        None,
    )
    beta_osf_me = np.loadtxt(ME_debias[0])
    fitt_osf_me = np.loadtxt(ME_debias[1])
    assert np.allclose(beta, beta_osf_me)
    assert np.allclose(fitt, fitt_osf_me)
