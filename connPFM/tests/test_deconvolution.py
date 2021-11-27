from os.path import join

import numpy as np


def test_stability_lars(surr_dir):
    """
    Test the SLARS computation
    """
    # Load data
    data_filename = join(surr_dir, "data.npy")
    data = np.load(data_filename)
    first = 0
    last = 0
    if first is None:
        nvoxels = 1
    else:
        nvoxels = last - first + 1
        voxel = first

    # Load HRF
    filename_hrf = join(surr_dir, "hrf.npy")
    hrf = np.load(filename_hrf)

    nvoxels = last - first + 1
    nscans = data.shape[0]
    auc = np.zeros((nscans, nvoxels))
    np.random.seed(200)
    from connPFM.deconvolution.stability_lars import StabilityLars

    sl = StabilityLars()
    for vox_idx in range(nvoxels):
        sl.stability_lars(hrf, np.expand_dims(data[:, voxel + vox_idx], axis=-1))
        auc[:, vox_idx] = np.squeeze(sl.auc)

    # load saved AUC
    auc_filename = join(surr_dir, "auc_0_OSF.npy")
    auc_osf = np.load(auc_filename)
    # Check if AUC is correct
    assert np.all(auc == auc_osf)
