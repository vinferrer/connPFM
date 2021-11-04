import os
import subprocess
from time import sleep

import nibabel as nib
import numpy as np


def transform(atlas_orig, data_tlrc, temp_dir):
    atlas_obj = nib.load(atlas_orig)
    data_obj = nib.load(data_tlrc)
    TMP_name = "atlas.nii.gz"
    if np.any(atlas_obj.affine != data_obj.affine):
        tmp_image = nib.Nifti1Image(atlas_obj.get_fdata(), data_obj.affine, data_obj.header)

        nib.save(tmp_image, os.path.join(temp_dir, TMP_name))
        subprocess.run(
            f"3drefit -space TLRC -view tlrc {os.path.join(temp_dir, TMP_name)}",
            shell=True,
        )
    else:
        os.system(f"cp {atlas_orig} {os.path.join(temp_dir, TMP_name)}")
    return os.path.join(temp_dir, TMP_name)


def inverse_transform(data_tlrc, atlas_orig):
    atlas_obj = nib.load(atlas_orig)
    data_obj = nib.load(data_tlrc)
    if np.any(atlas_obj.affine != data_obj.affine):
        subprocess.run(
            f"3drefit -space ORIG -view orig {data_tlrc}",
            shell=True,
        )
        sleep(5)
        tmp_data = nib.Nifti1Image(data_obj.get_fdata(), data_obj.affine, data_obj.header)
        nib.save(tmp_data, data_tlrc)
