import sys

import numpy as np
from joblib import Parallel, delayed
from nilearn.input_data import NiftiLabelsMasker

from connPFM.cli.caps import _get_parser
from connPFM.connectivity.ev import circular_shift_randomization


def read_data(path, atlas):
    """
    Reads data from a given path and masker.
    """
    masker = NiftiLabelsMasker(
        labels_img=atlas,
        standardize=False,
        memory="nilearn_cache",
        strategy="mean",
    )
    data = masker.fit_transform(path)
    return data


def caps(data, atlas, output, nsur=100, njobs=1):
    data = read_data(data, atlas)

    # Get number of time points/nodes
    [t, n] = data.shape

    # Compute RSS on orignal data
    rss = np.sum(data ** 2, axis=1)

    # Initialize rssr array with zeros
    rssr = np.zeros((t, nsur))

    # Generate surrogates and calculare RSS on them
    results = Parallel(n_jobs=njobs, backend="multiprocessing")(
        delayed(circular_shift_randomization())(data, n, t) for irand in range(nsur)
    )

    for irand in range(nsur):
        rssr[:, irand] = results[irand][0]


def _main(argv=None):
    """
    Main function.
    """
    options = _get_parser().parse_args(argv)
    options = vars(options)
    caps(
        data=options["data"][0],
        atlas=options["atlas"][0],
        output=options["output"][0],
        nsur=options["nsur"],
        njobs=options["njobs"],
    )


if __name__ == "__main__":
    _main(sys.argv[1:])
