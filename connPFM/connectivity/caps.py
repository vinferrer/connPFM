import sys
from os.path import join

import numpy as np
from joblib import Parallel, delayed
from nilearn.input_data import NiftiLabelsMasker
from nilearn.masking import apply_mask

from connPFM.cli.caps import _get_parser
from connPFM.connectivity.ev import (
    circular_shift_randomization,
    find_significant_frames,
    calculate_p_value,
)
from connPFM.utils import atlas_mod


def read_data(path, mask, atlas=True):
    """
    Reads data from a given path and mask or atlas.
    """
    if atlas:
        masker = NiftiLabelsMasker(
            labels_img=mask,
            standardize=False,
            memory="nilearn_cache",
            strategy="mean",
        )
        data = masker.fit_transform(path)
    else:
        masker = None
        data = apply_mask(path, mask)
    return data, masker


def caps(data_path, atlas, mask, output, nsur=100, njobs=1):
    data, masker = read_data(data_path, atlas)
    roi_data, _ = read_data(data_path, mask, atlas=True)

    # Get number of time points/nodes
    [t, n] = roi_data.shape

    # Compute RSS on orignal data
    rss = np.sum(roi_data ** 2, axis=1)

    # Initialize rssr array with zeros
    rssr = np.zeros((t, nsur))

    # Generate surrogates and calculare RSS on them
    results = Parallel(n_jobs=njobs, backend="multiprocessing")(
        delayed(circular_shift_randomization())(data, n, t) for irand in range(nsur)
    )

    for irand in range(nsur):
        rssr[:, irand] = results[irand][0]

    rssr[0, :] = 0

    # Calculate p-value and find significant frames
    p = calculate_p_value(rss, rssr)
    idxpeak = find_significant_frames(p)

    # Get selected maps from whole brain data and calculate mean
    caps = data[idxpeak, :]
    caps_mean = np.mean(caps, axis=0)

    # Save CAPs results
    if masker:
        caps_mean_4d = masker.inverse_transform(caps_mean)
        caps_mean_4d.to_filename(output)
        atlas_mod.inverse_transform(output, data_path)


def _main(argv=None):
    """
    Main function.
    """
    options = _get_parser().parse_args(argv)
    options = vars(options)
    caps(
        data=options["data_path"][0],
        atlas=options["atlas"][0],
        output=options["output"][0],
        nsur=options["nsur"],
        njobs=options["njobs"],
    )


if __name__ == "__main__":
    _main(sys.argv[1:])
