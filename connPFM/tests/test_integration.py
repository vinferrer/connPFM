from os.path import join

import random 
import numpy as np
import pandas as pd
import pytest
import subprocess
from nilearn.input_data import NiftiLabelsMasker

@pytest.mark.integration_pfm
def test_integration_pfm(testpath,bold_file,atlas_file,AUC_file):
    auc_output=join(testpath,'auc_local.nii.gz')
    subprocess.call("export mode=integration_pfm && connPFM -i {} -a {} --AUC {} -tr 1 -u vferrer -job 0 -nsur 1 -w pfm".format(bold_file,atlas_file,auc_output),shell=True)
    masker = NiftiLabelsMasker(
        labels_img=atlas_file,
        standardize=False,
        strategy="mean",
        resampling_target=None,
    )
    # compare the AUC values
    auc_osf=masker.fit_transform(AUC_file)
    auc_local=masker.fit_transform(auc_output)
    np.all(auc_osf==auc_local)

    