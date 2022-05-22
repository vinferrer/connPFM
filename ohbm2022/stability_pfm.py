from os.path import join as opj

import numpy as np
from joblib import Parallel, delayed
from nilearn.input_data import NiftiLabelsMasker

from connPFM.debiasing.debiasing import debiasing

prj_dir = "/bcbl/home/public/PJMASK_2/connPFM_data/sub-001/ses-01"
auc_file = opj(prj_dir, "connPFM_echoes/sub-001_AUC_300.nii.gz")
temp_dir = opj(prj_dir, "temp_sub-001_300_echoes")
data = opj(prj_dir, "connPFM_echoes/pb06.sub-002ParkMabCm.denoised_no_censor.nii.gz")
atlas = opj(prj_dir, "func_preproc/Schaefer2018_300Parcels_17Networks_subcorticals_func.nii.gz")
surrprefix = opj(temp_dir, "surrogate_AUC_")
surrsufix = ""
prefix = opj(prj_dir,"connPFM_echoess/sub-001_300_echoes_only_pfm")
nsur = 100
tr = 1.5
matrix_out = opj(prj_dir, "connPFM_echoes/only_pfm_matrix.txt")

###############################################################################
# Code starts here


# Read AUC of surrogate dataset
def read_surrogate_auc(surrprefix, surrsufix, masker, irand):
    auc = np.nan_to_num(masker.fit_transform(surrprefix + str(irand) + surrsufix + ".nii.gz"))
    return auc


# Get threshold value from surrogate data
def calculate_threshold(auc, percentile=95):
    thr = np.percentile(auc, percentile)
    return thr


def main():
    # Masker
    masker = NiftiLabelsMasker(
        labels_img=atlas,
        standardize=False,
        strategy="mean",
    )

    # Read AUC data
    auc = masker.fit_transform(auc_file)

    # Read AUC of surrogates
    surr_auc = Parallel(n_jobs=10, backend="multiprocessing")(
        delayed(read_surrogate_auc)(surrprefix, surrsufix, masker, irand) for irand in range(nsur)
    )

    # Calculate threshold
    thr = calculate_threshold(surr_auc)

    # Threshold AUC
    auc[auc < thr] = 0
    np.savetxt(matrix_out, auc)
    # Debiasing with non-zero thresholded AUC
    # debiasing(data, atlas, auc, tr, temp_dir, prefix, "")


if __name__ == "__main__":
    main()
