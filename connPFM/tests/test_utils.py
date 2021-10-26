from os.path import join

import numpy as np

from nilearn.input_data import NiftiLabelsMasker


def test_surrogate_generator(bold_file, atlas_file, testpath, surrogate_200):
    masker = NiftiLabelsMasker(
        labels_img=atlas_file,
        standardize=False,
        memory=None,
        strategy="mean",
        resampling_target=None,
    )
    np.random.seed(200)

    from connPFM.utils import surrogate_generator
    surrogate = surrogate_generator.generate_surrogate(bold_file,
                                                       atlas_file,
                                                       join(testpath,
                                                            'generated_surrogate.nii.gz'))
    keeped_surrogate = masker.fit_transform(surrogate_200)
    assert np.all(np.isclose(surrogate, keeped_surrogate))
