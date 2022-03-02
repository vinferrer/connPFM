import logging
import subprocess

import numpy as np
from nilearn.input_data import NiftiLabelsMasker

from connPFM.utils import atlas_mod

LGR = logging.getLogger(__name__)


def load_data(data, atlas, n_echos=1):
    """
    Load and mask data with atlas using NiftiLabelsMasker.
    """
    # Initialize masker object
    masker = NiftiLabelsMasker(labels_img=atlas, standardize=False, strategy="mean")

    # If n_echos is 1 (single echo), mask and return data
    if n_echos == 1:
        # If data is a list, keep only first element
        if isinstance(data, list):
            data = data[0]
        data_masked = masker.fit_transform(data)
    else:
        # If n_echos is > 1 (multi-echo), mask each echo in data list separately and
        # concatenate the masked data.
        for echo_idx, echo in enumerate(data):
            if echo_idx == 0:
                data_masked = masker.fit_transform(echo)
            else:
                data_masked = np.concatenate((data_masked, masker.fit_transform(echo)), axis=0)

    return data_masked, masker


def save_img(data, output, masker, history_str):
    """
    Save data as Nifti image, and update header history.
    """
    # Transform data back to Nifti image
    data_img = masker.inverse_transform(data)

    # Save data as Nifti image
    data_img.to_filename(output)

    # Transform space of image
    atlas_mod.inverse_transform(output)

    LGR.info("Updating file history...")
    subprocess.run('3dNotes -h "' + history_str + '" ' + output, shell=True)
    LGR.info("File history updated.")
