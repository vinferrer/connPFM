import logging
import os

import numpy as np
from nilearn.input_data import NiftiLabelsMasker

from connPFM.utils import atlas_mod

LGR = logging.getLogger(__name__)


def splitext_(path):
    if len(path.split(".")) > 2:
        return path.split(".")[0], ".".join(path.split(".")[-2:])
    return os.path.splitext(path)


def generate_surrogate(data, atlas, output):
    """
    Generate surrogate data.

    Parameters
    ----------
    data : Niimg-like object
        Data to generate surrogates from.
    atlas : Niimg-like object
        Mask with ROIs.
    output : str
        Path where surrogate data should be saved.

    Returns
    -------
    surrogate : Niimg-like object
        Surrogate data.
    """
    # Mask data
    LGR.info("Masking data...")
    surrogate_masker = NiftiLabelsMasker(
        labels_img=atlas, standardize="psc", memory="nilearn_cache", strategy="mean"
    )
    data_masked = surrogate_masker.fit_transform(data)
    LGR.info("Data masked.")

    surrogate = np.zeros(data_masked.shape)
    nscans = data_masked.shape[0]
    nvoxels = data_masked.shape[1]
    np.random.seed(200)
    for iter_tc in range(nvoxels):
        # phase_signal is a time x 1 vector filled with random phase
        # information (in rad, from -pi to pi)
        random_signal = np.fft.fft(np.random.uniform(size=nscans), nscans)
        phase_signal = np.angle(random_signal)

        # We multiply the magnitude of the original data with random phase
        # information to generate surrogate data
        surrogate[:, iter_tc] = np.real(
            np.fft.ifft(
                np.exp(1j * phase_signal) * abs(np.fft.fft(data_masked[:, iter_tc].T, nscans)),
                nscans,
            )
        )

    surrogate_output = surrogate_masker.inverse_transform(surrogate)

    output_filename, _ = splitext_(output)

    surrogate_output.to_filename(f"{output_filename}.nii.gz")
    atlas_mod.inverse_transform(f"{output_filename}.nii.gz", data)
    return surrogate
