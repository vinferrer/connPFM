from os.path import join

import numpy as np
from nilearn.input_data import NiftiLabelsMasker

from connPFM.connectivity import ev


def test_calculate_ets(ets_auc_original_file, AUC_file, atlas_file):
    masker = NiftiLabelsMasker(
        labels_img=atlas_file,
        standardize=False,
        strategy="mean",
        resampling_target=None,
    )

    AUC_img = masker.fit_transform(AUC_file)
    ets_auc_orig = np.loadtxt(ets_auc_original_file)
    ets_auc_calc, u_vec, v_vec = ev.calculate_ets(AUC_img, AUC_img.shape[1])
    assert np.all(np.isclose(ets_auc_orig, ets_auc_calc))


def test_rss_surr(AUC_file, atlas_file, surr_dir, rssr_auc_file):
    masker = NiftiLabelsMasker(
        labels_img=atlas_file,
        standardize=False,
        strategy="mean",
    )

    AUC_img = masker.fit_transform(AUC_file)
    _, u, v = ev.calculate_ets(AUC_img, AUC_img.shape[1])
    rssr, _, _ = ev.rss_surr(AUC_img, u, v, join(surr_dir, "surrogate_AUC_"), "", masker, 0)
    rssr_auc = np.loadtxt(rssr_auc_file)
    assert np.all(np.isclose(rssr, rssr_auc))


def test_threshold_ets_matrix():
    dum_mat = np.ones((3, 3))
    dum_mat[2, 1] = 3
    dum_mat2 = np.zeros((3, 3))
    dum_mat2[2, 1] = 3
    th_dum = ev.threshold_ets_matrix(dum_mat, [2], 2)
    assert np.all(th_dum == dum_mat2)


def test_event_detection(
    AUC_file, atlas_file, surr_dir, ets_auc_original_file, ets_auc_denoised_file, rssr_auc_file
):
    (
        ets_auc,
        rss_auc,
        _,
        idxpeak_auc,
        _,
        _,
        ets_auc_denoised,
        _,
        _,
    ) = ev.event_detection(AUC_file, atlas_file, join(surr_dir, "surrogate_AUC_"), nsur=10)
    assert np.all(ets_auc == np.loadtxt(ets_auc_original_file))
    assert np.all(np.isclose(ets_auc_denoised, np.loadtxt(ets_auc_denoised_file)))
