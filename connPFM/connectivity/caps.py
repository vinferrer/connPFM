import sys

import numpy as np
from joblib import Parallel, delayed
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
from nilearn.masking import apply_mask

from connPFM.cli.caps import _get_parser
from connPFM.connectivity.ev import (
    circular_shift_randomization,
    find_significant_frames,
    calculate_pvalue,
)
from connPFM.utils import atlas_mod


def read_data(path, mask, atlas=True,standard="psc"):
    """
    Reads data from a given path and mask or atlas.
    """
    if atlas:
        masker = NiftiLabelsMasker(
            labels_img=mask,
            standardize=standard,
            memory="nilearn_cache",
            strategy="mean",
        )
        data = masker.fit_transform(path)
        return data, masker
    else:
        masker = NiftiMasker(
            mask_img=mask,
            standardize=standard,
            memory="nilearn_cache",
        )
        data = masker.fit_transform(path)
        return data


def caps(data_path, atlas, mask, output,surrprefix =None,surrsufix = "", nsur=100, njobs=-1):
    # Read original data and data inside ROI
    print("Reading data...")
    data, masker = read_data(data_path, atlas)
    roi_data = read_data(data_path, mask, atlas=False)

    # Get number of time points/nodes
    [t, n] = roi_data.shape

    # Compute RSS on orignal data
    print("Calculating RSS of ROI data...")
    rss = np.sqrt(np.sum(roi_data ** 2, axis=1))

    # Initialize rssr array with zeros
    rssr = np.zeros((t, nsur))

    # Generate surrogates and calculare RSS on them
    if surrprefix is None:
        print("Generating surrogates of ROI data...")
        results = Parallel(n_jobs=njobs, backend="multiprocessing")(
            delayed(circular_shift_randomization)(roi_data, n, t) for irand in range(nsur)
        )
    else:
        print("Reading surrogates...")
        results = Parallel(n_jobs=njobs, backend="multiprocessing")(
            delayed(read_data)(surrprefix + str(irand) + surrsufix + ".nii.gz",mask,False,False) for irand in range(nsur)
        )
    
    # Calculate RSS of surrogate data
    print("Calculating RSS of surrogate data...")
    for irand in range(nsur):
        rssr[:, irand] = np.sqrt(np.sum(results[irand] ** 2, axis=1))

    rssr[0, :] = 0

    # Calculate p-value and find significant frames
    print("Finding significant peaks...")
    p_value = calculate_pvalue(rss, rssr)
    idxpeak = find_significant_frames(p_value)

    # Get selected maps from whole brain data and calculate mean
    print("Calculating average map of selected indices...")
    caps = data[idxpeak, :]
    caps_mean = np.expand_dims(np.mean(caps, axis=0), axis=0)

    # Save CAPs results
    print("Saving CAPs map...")
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
        data_path=options["data_path"],
        atlas=options["atlas"],
        output=options["output"],
        mask=options["mask"],
        surrprefix=options["surrprefix"],
        nsur=options["nsur"],
        njobs=options["njobs"],
    )


if __name__ == "__main__":
    _main(sys.argv[1:])
