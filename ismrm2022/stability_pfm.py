import numpy as np
from joblib import Parallel, delayed
from nilearn.input_data import NiftiLabelsMasker
from os.path import join as opj

from connPFM.debiasing.debiasing import debiasing

prj_dir = "/export/home/eurunuela/public/PARK_VFERRER/toolbox_data/sub-002ParkMabCm_100"
auc_file = opj(prj_dir, "sub-002ParkMabCm_AUC_100.nii.gz")
temp_dir = opj(prj_dir, "temp_sub-002ParkMabCm_100")
data = opj(prj_dir, "pb06.sub-002ParkMabCm.denoised_no_censor.nii.gz")
atlas = opj(temp_dir, "atlas.nii.gz")
surrprefix = opj(temp_dir, "surrogate_AUC_")
surrsufix = ""
nsur = 100
tr = 1

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
        memory="",
        strategy="mean",
    )

    # Read AUC data
    auc = masker.fit_transform(auc_file)

    # Read AUC of surrogates
    surr_auc = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(read_surrogate_auc)(surrprefix, surrsufix, masker, irand) for irand in range(nsur)
    )

    # Calculate threshold
    thr = calculate_threshold(surr_auc)

    # Threshold AUC
    auc[auc < thr] = 0

    # Debiasing with non-zero thresholded AUC
    debiasing(data, atlas, auc, tr, temp_dir, "")


if __name__ == "__main__":
    main()
