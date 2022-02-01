"""Main debiasing workflow."""
import logging
import subprocess
from os.path import join

import numpy as np
from nilearn.input_data import NiftiLabelsMasker

from connPFM.debiasing.debiasing_functions import debiasing_spike  # or debiasing_block
from connPFM.utils import atlas_mod
from connPFM.utils.hrf_generator import HRFMatrix

LGR = logging.getLogger(__name__)


def debiasing(data_file, mask, mtx, tr, out_dir, prefix, groups, groups_dist, history_str):
    """Perform debiasing based on denoised edge-time matrix."""
    LGR.info("Performing debiasing based on denoised edge-time matrix...")
    masker = NiftiLabelsMasker(
        labels_img=mask,
        standardize=False,
        strategy="mean",
    )

    # Read data
    data = masker.fit_transform(data_file)

    # Get number of time points/nodes
    [_, n] = data.shape

    # Get ETS indexes
    idx_u, idx_v = np.argwhere(np.triu(np.ones(n), 1)).T

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
    deb_output = debiasing_spike(hrf, data, ets_mask, groups=groups, group_dist=groups_dist)
    beta = deb_output["beta"]
    fitt = deb_output["betafitts"]

    # Transform results back to 4D
    beta_4D = masker.inverse_transform(beta)
    beta_file = join(out_dir, f"{prefix}_beta_ETS.nii.gz")
    beta_4D.to_filename(beta_file)
    atlas_mod.inverse_transform(beta_file)
    subprocess.run(f"3dNotes {beta_file} -h {history_str}", shell=True)

    fitt_4D = masker.inverse_transform(fitt)
    fitt_file = join(out_dir, f"{prefix}_fitt_ETS.nii.gz")
    fitt_4D.to_filename(fitt_file)
    subprocess.run(f"3dNotes {fitt_file} -h {history_str}", shell=True)
    atlas_mod.inverse_transform(fitt_file)

    LGR.info("Debiasing finished and files saved.")

    return beta, fitt
