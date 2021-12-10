import subprocess
from os.path import basename, dirname, join

import numpy as np
import pytest
from nilearn.input_data import NiftiLabelsMasker


def test_integration_pfm(testpath, bold_file, atlas_1roi, AUC_file, skip_integration):
    if skip_integration:
        pytest.skip("Skipping integration test")
    auc_output = join(testpath, "auc_local.nii.gz")
    subprocess.call(
        "export mode=integration_pfm && "
        "connPFM -i {} -a {} --AUC {} -d {} -tr 1 -u vferrer -job 0 -nsur 1 -w pfm".format(
            bold_file, atlas_1roi, auc_output, join(testpath, "temp")
        ),
        shell=True,
    )
    masker = NiftiLabelsMasker(
        labels_img=atlas_1roi,
        standardize=False,
        strategy="mean",
        resampling_target=None,
    )
    # compare the AUC values
    auc_osf = masker.fit_transform(AUC_file)
    auc_local = masker.fit_transform(auc_output)
    np.allclose(auc_osf, auc_local)


def test_integration_ev(
    testpath,
    bold_file,
    atlas_file,
    AUC_file,
    ets_auc_denoised_file,
    surr_dir,
    skip_integration,
    ets_rss_thr_file,
):
    if skip_integration:
        pytest.skip("Skipping integration test")

    subprocess.call(
        "connPFM -i {} -a {} --AUC {} -d {} -m {} ".format(
            bold_file,
            atlas_file,
            AUC_file,
            surr_dir,
            join(dirname(AUC_file), "ets_AUC_denoised.txt"),
        )
        + "--peaks_points ets_AUC_denoised -tr 1 -u vferrer -nsur 50 -w ev",
        shell=True,
    )
    ets_auc_denoised_local = np.loadtxt(join(dirname(AUC_file), "ets_AUC_denoised.txt"))
    ets_auc_osf = np.loadtxt(join(ets_auc_denoised_file))
    rss_out_local = np.loadtxt(join(testpath, "ets_AUC_denoised_rss_th.txt"))
    rss_out_osf = np.loadtxt(ets_rss_thr_file)
    np.allclose(ets_auc_denoised_local, ets_auc_osf)
    np.allclose(rss_out_local, rss_out_osf)


def test_integration_debias(
    testpath,
    bold_file,
    atlas_file,
    AUC_file,
    ets_auc_denoised_file,
    surr_dir,
    beta_file,
    fitt_file,
    skip_integration,
):
    if skip_integration:
        pytest.skip("Skipping integration test")
    subprocess.call(
        "connPFM -i {} -a {} --AUC {} -d {} -m {} --prefix {} ".format(
            bold_file,
            atlas_file,
            AUC_file,
            surr_dir,
            ets_auc_denoised_file,
            f"{basename(bold_file[:-7])}",
        )
        + "-tr 1 -u vferrer -nsur 50 -w debias",
        shell=True,
    )
    masker = NiftiLabelsMasker(
        labels_img=atlas_file,
        standardize=False,
        strategy="mean",
    )
    beta_osf = masker.fit_transform(beta_file)
    fitt_osf = masker.fit_transform(fitt_file)
    beta_local = masker.fit_transform(
        join(testpath, f"{basename(bold_file[:-7])}_beta_ETS.nii.gz")
    )
    fitt_local = masker.fit_transform(
        join(testpath, f"{basename(bold_file[:-7])}_fitt_ETS.nii.gz")
    )

    np.allclose(beta_osf, beta_local)
    np.allclose(fitt_osf, fitt_local)
