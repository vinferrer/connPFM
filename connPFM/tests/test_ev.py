from os.path import isfile, join

import numpy as np
from nilearn.input_data import NiftiLabelsMasker

from connPFM.connectivity import connectivity_utils, ev, plotting


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
    nbins = 1000

    AUC_img = masker.fit_transform(AUC_file)
    _, u, v = connectivity_utils.calculate_ets(AUC_img, AUC_img.shape[1])
    rssr, _, _, _ = connectivity_utils.rss_surr(
        AUC_img, u, v, join(surr_dir, "surrogate_AUC_"), "", masker, 0, nbins
    )
    rssr_auc = np.loadtxt(rssr_auc_file)
    assert np.all(np.isclose(rssr, rssr_auc))


def test_threshold_ets_matrix():
    dum_mat = np.ones((3, 3))
    dum_mat[2, 1] = 3
    dum_mat2 = np.zeros((3, 3))
    dum_mat2[2, 1] = 1
    th_dum = connectivity_utils.threshold_ets_matrix(dum_mat, thr=2, selected_idxs=2)
    assert np.allclose(th_dum, dum_mat2)


def test_calculate_surrogate_ets(atlas_file, surr_dir, surrogate_ets_file):
    masker = NiftiLabelsMasker(
        labels_img=atlas_file,
        standardize=False,
        strategy="mean",
    )
    # Test surrogate ets calculation
    ets_surr = connectivity_utils.calculate_surrogate_ets(
        surrprefix=join(surr_dir, "surrogate_AUC_"), sursufix="", irand=0, masker=masker
    )
    assert np.allclose(ets_surr, np.load(surrogate_ets_file))


def test_calculate_hist(atlas_file, surr_dir, surrogate_hist_file):
    masker = NiftiLabelsMasker(
        labels_img=atlas_file,
        standardize=False,
        strategy="mean",
    )
    # Test calculate_hist
    hist, _ = connectivity_utils.calculate_hist(
        surrprefix=join(surr_dir, "surrogate_AUC_"),
        sursufix="",
        irand=0,
        masker=masker,
        hist_range=(0, 1),
    )
    assert np.allclose(hist, np.load(surrogate_hist_file))


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


def test_plotting(testpath, ets_auc_denoised_all):
    # Test plotting only to improve coverage
    ets = np.load(ets_auc_denoised_all)[:, :, 3]
    rss = np.random.uniform(0, 1, ets.shape[1])
    dummy_enorm_file = join(testpath, "dummy_enorm.txt")
    np.savetxt(dummy_enorm_file, rss)
    plotting.plot_ets_matrix(ets, testpath, rss, sufix="_rss")
    assert isfile(join(testpath, "ets_rss.png"))
    plotting.plot_ets_matrix(
        ets,
        testpath,
        rss,
        sufix="_enorm",
        dvars_file=dummy_enorm_file,
        enorm_file=dummy_enorm_file,
    )
    assert isfile(join(testpath, "ets_enorm.png"))
