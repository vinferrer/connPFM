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
        "x6r8v", testpath, "sub-pixar123_task-pixar_space-MNI152-preproc_bold.nii.gz"
    )


@pytest.fixture
def AUC_file(testpath):
    return fetch_file("h6uv3", testpath, "sub-pixar123_task-pixar_AUC_100.nii.gz")


@pytest.fixture
def ets_auc_original_file(testpath):
    return fetch_file("bnp4z", testpath, "ets_AUC_original.txt")


@pytest.fixture
def ets_auc_denoised_file(testpath):
    return fetch_file("jvuwn", testpath, "ets_AUC_denoised.txt")


@pytest.fixture
def surr_dir(testpath):
    zipped_file = fetch_file("yrkqv", testpath, "temp_sub-pixar123_100.tar.gz")
    my_tar = tarfile.open(zipped_file)
    my_tar.extractall(testpath)  # specify which folder to extract to
    my_tar.close()
    return os.path.join(testpath, "temp_sub-pixar123_100")


@pytest.fixture
def atlas_file(testpath):
    return fetch_file("bhcrp", testpath,
                      "Schaefer2018_100Parcels_17Networks_order_FSLMNI152_4mm.nii.gz")


@pytest.fixture
def rssr_auc_file(testpath):
    return fetch_file("7pnj2", testpath, "rssr_AUC.txt")
