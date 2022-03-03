"""Main debiasing workflow."""
import logging
from os.path import join

import numpy as np

from connPFM.debiasing.debiasing_functions import debiasing_spike  # or debiasing_block
from connPFM.utils import io
from connPFM.utils.hrf_generator import HRFMatrix

LGR = logging.getLogger(__name__)


def debiasing(data_file, mask, te, mtx, tr, out_dir, prefix, groups, groups_dist, history_str):
    """Perform debiasing based on denoised edge-time matrix."""
    if te is None:
        te = [0]
    elif len(te) > 1:
        # If all values in TE list are higher than 1, divide them by 1000.
        # Only relevant for multi-echo data.
        if all(te_val > 1 for te_val in te):
            te = [te_val / 1000 for te_val in te]

    LGR.info("Performing debiasing based on denoised edge-time matrix...")
    # Read data
    data, masker = io.load_data(data_file, mask, n_echos=len(te))

    # Get number of nodes
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
    beta_file = join(out_dir, f"{prefix}_beta_ETS.nii.gz")
    io.save_img(beta, beta_file, masker, history_str)

    fitt_file = join(out_dir, f"{prefix}_fitt_ETS.nii.gz")
    io.save_img(fitt, fitt_file, masker, history_str)

    LGR.info("Debiasing finished and files saved.")

    return beta, fitt
