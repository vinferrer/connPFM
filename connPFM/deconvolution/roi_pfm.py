import datetime
import getpass
import os
import socket
import subprocess

import numpy as np
from nilearn.input_data import NiftiLabelsMasker

from utils import atlas_mod
from cli import _get_parser
from utils.hrf_matrix import HRFMatrix
from deconvolution.run_stability_lars_bcbl import run_stability_lars


def splitext_(path):
    if len(path.split(".")) > 2:
        return path.split(".")[0], ".".join(path.split(".")[-2:])
    return os.path.splitext(path)


def generate_surrogate(data, atlas, atlas_orig, output):
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
    print("Masking data...")
    surrogate_masker = NiftiLabelsMasker(
        labels_img=atlas, standardize="psc", memory="nilearn_cache", strategy="mean"
    )
    data_masked = surrogate_masker.fit_transform(data)
    print("Data masked.")

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

    surrogate_output = surrogate_masker.inverse_transform(surrogate)

    output_filename, _ = splitext_(output)

    surrogate_output.to_filename(f"{output_filename}.nii.gz")
    atlas_mod.inverse_transform(f"{output_filename}.nii.gz", data)
    return surrogate


def _main(argv=None):
    """Entry point for ROI PFM."""
    # Command to save in file's history to be seen with the 3dinfo command
    options = _get_parser().parse_args(argv)

    args_str = str(options)[9:]
    history_str = "[{username}@{hostname}: {date}] python debiasing.py with {arguments}".format(
        username=getpass.getuser(),
        hostname=socket.gethostname(),
        date=datetime.datetime.now().strftime("%c"),
        arguments=args_str,
    )

    kwargs = vars(options)
    kwargs["history"] = history_str

    ####################
    #    EDIT BELOW    #
    ####################

    # Use full path or the os.path.join() function
    data = kwargs["data"][0]
    temp_dir = kwargs["dir"]
    output_file = kwargs["output"][0]

    # Choose one of fetch_atlas_XXXX functions
    # https://nilearn.github.io/modules/reference.html#module-nilearn.datasets
    # You can also use the path to a local atlas file.
    atlas = kwargs["atlas"][0]
    # For HRF
    if kwargs["te"] is not None:
        te = kwargs["te"][0]  # Use any number for single-echo. Use ms for multi-echo.
    else:
        te = kwargs["te"]
    tr = kwargs["tr"][0]
    # Use AFNI generated HRF (3dDeconvolve)
    is_afni = True
    lop_hrf = "SPMG1"  # Default is SPMG1
    hrf_path = None  # For custom HRF
    # True for block model
    integrator = kwargs["block"]

    # For stability selection with LARS
    # Number of surrogates for stability selection
    n_stability_surrogates = 50
    maxiterfactor = 0.8

    # For HPC cluster
    # Number of parallel jobs to send (splits in groups of voxels)
    jobs = kwargs["jobs"][0]
    # HPC username to check running jobs
    username = kwargs["username"][0]

    n_auc_surrogates = kwargs["nsurrogates"][0]

    ########################
    # DO NOT  EDIT BELOW   #
    ########################
    os.makedirs(temp_dir, exist_ok=True)
    print("Masking data...")
    atlas_old = atlas
    atlas = atlas_mod.transform(atlas, data, temp_dir)
    masker = NiftiLabelsMasker(
        labels_img=atlas, standardize="psc", memory="nilearn_cache", strategy="mean"
    )
    data_masked = masker.fit_transform(data)
    print("Data masked.")
    print("Generating HRF...")
    # Generates HRF matrix
    hrf_matrix = HRFMatrix(
        TR=tr,
        TE=te,
        nscans=data_masked.shape[0],
        r2only=True,
        has_integrator=integrator,
        is_afni=is_afni,
        lop_hrf=lop_hrf,
        path=hrf_path,
    )
    hrf_matrix.generate_hrf()
    hrf = hrf_matrix.hrf_norm
    print("HRF generated.")

    print("Running stability selection on original data...")
    auc = run_stability_lars(
        data=data_masked,
        hrf=hrf,
        temp=temp_dir,
        jobs=jobs,
        username=username,
        niter=n_stability_surrogates,
        maxiterfactor=maxiterfactor,
    )
    print("Stability selection on original data finished.")

    print("Saving AUC results of original data...")
    auc_4d = masker.inverse_transform(auc)
    auc_4d.to_filename(output_file)
    atlas_mod.inverse_transform(output_file, data)
    print("AUC results on original data saved.")

    print("Updating file history...")
    subprocess.run('3dNotes -h "' + history_str + '" ' + output_file, shell=True)
    print("File history updated.")

    if n_auc_surrogates:
        print(f"Performing PFM on {n_auc_surrogates} surrogates...")
        print("Make yourself a cup of coffee while it runs :)")
        for n_sur in range(n_auc_surrogates):
            # Generate surrogate
            surrogate_name = os.path.join(temp_dir, f"surrogate_{n_sur}.nii.gz")
            surrogate_masked = generate_surrogate(
                data=data, atlas=atlas, atlas_orig=atlas_old, output=surrogate_name
            )
            # Calculate AUC
            auc = run_stability_lars(
                data=surrogate_masked,
                hrf=hrf,
                temp=temp_dir,
                jobs=jobs,
                username=username,
                niter=n_stability_surrogates,
                maxiterfactor=maxiterfactor,
            )

            # Transform back to 4D
            auc_4d = masker.inverse_transform(auc)

            # Save surrogate AUC
            surrogate_out = os.path.join(temp_dir, f"surrogate_AUC_{n_sur}.nii.gz")
            auc_4d.to_filename(surrogate_out)
            atlas_mod.inverse_transform(surrogate_out, data)
            print(f"{n_sur}/{n_auc_surrogates -1 }")

        print(f"PFM on {n_auc_surrogates} surrogates finished.")

    print("PFM finished.")


if __name__ == "__main__":
    _main()
