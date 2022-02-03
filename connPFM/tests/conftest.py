import os
import ssl
import tarfile
from urllib.request import urlretrieve

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skipintegration", action="store_true", default=False, help="Skip integration tests."
    )


@pytest.fixture
def skip_integration(request):
    return request.config.getoption("--skipintegration")


def fetch_file(osf_id, path, filename):
    """
    Fetches file located on OSF and downloads to `path`/`filename`1
    Parameters
    ----------
    osf_id : str
        Unique OSF ID for file to be downloaded. Will be inserted into relevant
        location in URL: https://osf.io/{osf_id}/download
    path : str
        Path to which `filename` should be downloaded. Ideally a temporary
        directory
    filename : str
        Name of file to be downloaded (does not necessarily have to match name
        of file on OSF)
    Returns
    -------
    full_path : str
        Full path to downloaded `filename`
    """
    # This restores the same behavior as before.
    # this three lines make tests dowloads work in windows
    if os.name == "nt":
        orig_sslsocket_init = ssl.SSLSocket.__init__
        ssl.SSLSocket.__init__ = (
            lambda *args, cert_reqs=ssl.CERT_NONE, **kwargs: orig_sslsocket_init(
                *args, cert_reqs=ssl.CERT_NONE, **kwargs
            )
        )
        ssl._create_default_https_context = ssl._create_unverified_context
    url = "https://osf.io/{}/download".format(osf_id)
    full_path = os.path.join(path, filename)
    if not os.path.isfile(full_path):
        urlretrieve(url, full_path)
    return full_path


@pytest.fixture(scope="session")
def testpath(tmp_path_factory):
    """Test path that will be used to download all files"""
    return tmp_path_factory.getbasetemp()


@pytest.fixture
def bold_file(testpath):
    return fetch_file(
        "y3wa9", testpath, "sub-pixar123_task-pixar_space-MNI152-preproc_bold.SPC.nii.gz"
    )


@pytest.fixture
def AUC_file(testpath):
    return fetch_file("nh6y9", testpath, "sub-pixar123_task-pixar_AUC_100_200_seed.nii.gz")


@pytest.fixture
def ets_auc_original_file(testpath):
    return fetch_file("56apy", testpath, "ets_AUC_original_200_seed.txt")


@pytest.fixture
def ets_auc_denoised_file(testpath):
    return fetch_file("j96my", testpath, "ets_AUC_denoised_200_seed.txt")


@pytest.fixture
def surr_dir(testpath):
    zipped_file = fetch_file("yrkqv", testpath, "temp_sub-pixar123_100.tar.gz")
    my_tar = tarfile.open(zipped_file)
    my_tar.extractall(testpath)  # specify which folder to extract to
    my_tar.close()
    return os.path.join(testpath, "temp_sub-pixar123_100")


@pytest.fixture
def atlas_file(testpath):
    return fetch_file(
        "bhcrp", testpath, "Schaefer2018_100Parcels_17Networks_order_FSLMNI152_4mm.nii.gz"
    )


@pytest.fixture
def atlas_1roi(testpath):
    return fetch_file("ankvr", testpath, "Schaefer_1Parcel_4mm.nii.gz")


@pytest.fixture
def rssr_auc_file(testpath):
    return fetch_file("7pnj2", testpath, "rssr_AUC.txt")


@pytest.fixture
def surrogate_200(testpath):
    return fetch_file("wxfz7", testpath, "random_200_surrogate.nii.gz")


@pytest.fixture
def hrf_file(testpath):
    return fetch_file("gefu4", testpath, "hrf.txt")


@pytest.fixture
def hrf_linear_file(testpath):
    return fetch_file("mkeu2", testpath, "hrf_linear.txt")


@pytest.fixture
def beta_file(testpath):
    return fetch_file("2pmju", testpath, "sub-pixar123_task-pixar_beta_ETS.nii.gz")


@pytest.fixture
def fitt_file(testpath):
    return fetch_file("ud369", testpath, "sub-pixar123_task-pixar_fitt_ETS.nii.gz")


@pytest.fixture
def beta_block_file(testpath):
    return fetch_file("jhv7a", testpath, "beta_block.txt")


@pytest.fixture
def fitt_group_file(testpath):
    return fetch_file("nuceq", testpath, "betafitt_spike_group.txt")


@pytest.fixture
def ets_auc_all(testpath):
    return fetch_file("dqyxs", testpath, "ets_AUC_all.npy")


@pytest.fixture
def ets_auc_denoised_all(testpath):
    return fetch_file("c6xq4", testpath, "ets_AUC_denoised_all.npy")


@pytest.fixture
def surrogate_ets_file(testpath):
    return fetch_file("sqrce", testpath, "surrogate_ets.npy")


@pytest.fixture
def surrogate_hist_file(testpath):
    return fetch_file("3b6ge", testpath, "surrogate_hist.npy")


@pytest.fixture
def ets_rss_thr_file(testpath):
    return fetch_file("gn8x5", testpath, "ets_AUC_denoised_rss_th.txt")
