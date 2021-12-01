from os.path import join

import numpy as np
from nilearn.input_data import NiftiLabelsMasker

from connPFM.connectivity import connectivity_utils, ev


def test_calculate_ets(ets_auc_original_file, AUC_file, atlas_file):
    masker = NiftiLabelsMasker(
        labels_img=atlas_file,
        standardize=False,
        strategy="mean",
        resampling_target=None,
    )

    AUC_img = masker.fit_transform(AUC_file)
    ets_auc_orig = np.loadtxt(ets_auc_original_file)
    ets_auc_calc, u_vec, v_vec = connectivity_utils.calculate_ets(AUC_img, AUC_img.shape[1])
    assert np.all(np.isclose(ets_auc_orig, ets_auc_calc))


def test_rss_surr(AUC_file, atlas_file, surr_dir, rssr_auc_file):
    masker = NiftiLabelsMasker(
        labels_img=atlas_file,
        standardize=False,
        strategy="mean",
    )

    AUC_img = masker.fit_transform(AUC_file)
    _, u, v = connectivity_utils.calculate_ets(AUC_img, AUC_img.shape[1])
    rssr, _, _ = connectivity_utils.rss_surr(
        AUC_img, u, v, join(surr_dir, "surrogate_AUC_"), "", masker, 0
    )
    rssr_auc = np.loadtxt(rssr_auc_file)
    assert np.all(np.isclose(rssr, rssr_auc))


def test_threshold_ets_matrix():
    thr = 2
    dum_mat = np.ones((3, 3))
    dum_mat[2, 1] = 3
    dum_mat2 = np.zeros((3, 3))
    dum_mat2[2, 1] = 3 - thr
    th_dum = connectivity_utils.threshold_ets_matrix(dum_mat, thr=thr, selected_idxs=2)
    assert np.allclose(th_dum, dum_mat2)


def test_event_detection_rss(
    AUC_file, atlas_file, surr_dir, ets_auc_original_file, ets_auc_denoised_file
):
    # Test event detection with RSS option
    (ets_rss, _, _, _, ets_denoised_rss, _, _, _,) = ev.event_detection(
        data_file=AUC_file,
        atlas=atlas_file,
        surrprefix=join(surr_dir, "surrogate_AUC_"),
        nsur=10,
        peak_detection="rss",
    )
    assert np.allclose(ets_rss, np.loadtxt(ets_auc_original_file))
    assert np.allclose(ets_denoised_rss, np.loadtxt(ets_auc_denoised_file))


def test_event_detection_rss_time(
    AUC_file, atlas_file, surr_dir, ets_auc_all, ets_auc_denoised_all
):
    # Test event detection with RSS_time option
    (ets_rss_time, _, _, _, ets_denoised_rss_time, _, _, _,) = ev.event_detection(
        data_file=AUC_file,
        atlas=atlas_file,
        surrprefix=join(surr_dir, "surrogate_AUC_"),
        nsur=10,
        peak_detection="rss_time",
    )

    assert np.allclose(ets_rss_time, np.load(ets_auc_all)[:, :, 1])
    assert np.allclose(ets_denoised_rss_time, np.load(ets_auc_denoised_all)[:, :, 1])


def test_event_detection_ets(AUC_file, atlas_file, surr_dir, ets_auc_all, ets_auc_denoised_all):
    # Test event detection with ETS option
    (ets, _, _, _, ets_denoised, _, _, _,) = ev.event_detection(
        data_file=AUC_file,
        atlas=atlas_file,
        surrprefix=join(surr_dir, "surrogate_AUC_"),
        nsur=10,
        peak_detection="ets",
    )

    assert np.allclose(ets, np.load(ets_auc_all)[:, :, 2])
    assert np.allclose(ets_denoised, np.load(ets_auc_denoised_all)[:, :, 2])


def test_event_detection_ets_time(
    AUC_file, atlas_file, surr_dir, ets_auc_all, ets_auc_denoised_all
):
    # Test event detection with ETS_time option
    (ets_time, _, _, _, ets_denoised_time, _, _, _,) = ev.event_detection(
        data_file=AUC_file,
        atlas=atlas_file,
        surrprefix=join(surr_dir, "surrogate_AUC_"),
        nsur=10,
        peak_detection="ets_time",
    )

    assert np.allclose(ets_time, np.load(ets_auc_all)[:, :, 3])
    assert np.allclose(ets_denoised_time, np.load(ets_auc_denoised_all)[:, :, 3])
