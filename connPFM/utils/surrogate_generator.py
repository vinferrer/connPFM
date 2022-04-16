import logging
import os

import numpy as np

from connPFM.utils import io

LGR = logging.getLogger(__name__)


def splitext_(path):
    if len(path.split(".")) > 2:
        return path.split(".")[0], ".".join(path.split(".")[-2:])
    return os.path.splitext(path)


def generate_surrogate(data_masked, surrogate_masker, output):
    """
    Generate surrogate data.

    Parameters
    ----------
    data : numy.ndarray
        Data to generate surrogates from.
    atlas : Niimg-like object
        masker to save the data
    output : str
        Path where surrogate data should be saved.

    Returns
    -------
    surrogate : Niimg-like object
        Surrogate data.
    """

    surrogate = np.zeros(data_masked.shape)
    nscans = data_masked.shape[0]
    nvoxels = data_masked.shape[1]
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

    output_filename, _ = splitext_(output)
    io.save_img(surrogate, f"{output_filename}.nii.gz", surrogate_masker)

    return surrogate
