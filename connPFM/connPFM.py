import datetime
import getpass
import logging
import os
import socket
import subprocess
import sys

from nilearn.input_data import NiftiLabelsMasker
from numpy import loadtxt

from connectivity import ev
from cli.connPFM import _get_parser
from deconvolution.stability_lars_caller import run_stability_lars
from utils import atlas_mod, hrf_generator, surrogate_generator

LGR = logging.getLogger(__name__)
LGR.setLevel(logging.INFO)


def connPFM(
    data,
    atlas,
    output,
    tr,
    username,
    te=None,
    dir="temp",
    block=False,
    jobs=50,
    nsurrogates=50,
    nstability=50,
    percentile=95,
    maxiterfactor=0.3,
    hrf_shape="SPMG1",
    hrf_path=None,
    history_str=''
):

    if te is None:
        te = [0]

    os.makedirs(dir, exist_ok=True)

    # TODO: make it multi-echo compatible
    LGR.info("Masking data...")
    atlas_old = atlas
    atlas = atlas_mod.transform(atlas, data, dir)
    masker = NiftiLabelsMasker(
        labels_img=atlas, standardize="psc", memory="nilearn_cache", strategy="mean"
    )
    data_masked = masker.fit_transform(data)
    LGR.info("Data masked.")

    LGR.info("Generating HRF...")
    # Generates HRF matrix
    hrf_matrix = hrf_generator.HRFMatrix(
        TR=tr,
        TE=te,
        nscans=data_masked.shape[0],
        r2only=True,
        block=block,
        is_afni=True,
        lop_hrf=hrf_shape,
        path=hrf_path,
    )
    hrf_matrix.generate_hrf()
    hrf = hrf_matrix.hrf_norm
    LGR.info("HRF generated.")

    LGR.info("Running stability selection on original data...")
    auc = run_stability_lars(
        data=data_masked,
        hrf=hrf,
        temp=dir,
        jobs=jobs,
        username=username,
        niter=nstability,
        maxiterfactor=maxiterfactor,
    )
    LGR.info("Stability selection on original data finished.")

    LGR.info("Saving AUC results of original data...")
    auc_4d = masker.inverse_transform(auc)
    auc_4d.to_filename(output)
    atlas_mod.inverse_transform(output, data)
    LGR.info("AUC results on original data saved.")

    LGR.info("Updating file history...")
    subprocess.run('3dNotes -h "' + history_str + '" ' + output, shell=True)
    LGR.info("File history updated.")

    if nsurrogates:
        LGR.info(f"Performing PFM on {nsurrogates} surrogates...")
        LGR.info("Make yourself a cup of coffee while it runs :)")
        for n_sur in range(nsurrogates):
            # Generate surrogate
            surrogate_name = os.path.join(dir, f"surrogate_{n_sur}.nii.gz")
            surrogate_masked = surrogate_generator.generate_surrogate(
                data=data, atlas=atlas, atlas_orig=atlas_old, output=surrogate_name
            )
            # Calculate AUC
            auc = run_stability_lars(
                data=surrogate_masked,
                hrf=hrf,
                temp=dir,
                jobs=jobs,
                username=username,
                niter=nstability,
                maxiterfactor=maxiterfactor,
            )

            # Transform back to 4D
            auc_4d = masker.inverse_transform(auc)

            # Save surrogate AUC
            surrogate_out = os.path.join(dir, f"surrogate_AUC_{n_sur}.nii.gz")
            auc_4d.to_filename(surrogate_out)
            atlas_mod.inverse_transform(surrogate_out, data)
            LGR.info(f"{n_sur}/{nsurrogates -1 }")

        LGR.info(f"PFM on {nsurrogates} surrogates finished.")

    LGR.info("PFM finished.")


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    options = vars(options)
    args_str = str(options)[9:]
    history_str = "[{username}@{hostname}: {date}] python debiasing.py with {arguments}".format(
        username=getpass.getuser(),
        hostname=socket.gethostname(),
        date=datetime.datetime.now().strftime("%c"),
        arguments=args_str,
    )
    if options['workflow'][0] == 'all':
        connPFM(options['data'][0],
                options['atlas'][0],
                options['auc'][0],
                options['tr'][0],
                options['username'][0],
                options['te'],
                options['dir'],
                options['block'],
                options['jobs'][0],
                options['nsurrogates'][0],
                options['nstability'],
                options['percentile'],
                options['maxiterfactor'],
                options['hrf_shape'],
                options['hrf_path'],
                history_str)
        
        ev.ev_workflow(options['data'][0],
                    options['auc'][0],
                    options['atlas'][0],
                    options['nsurrogates'],
                    os.path.abspath(options['auc'][0]),
                    history_str)
        LGR.info("Perform debiasing based on edge-time matrix.")
        ev.debiasing(options['data'][0], 
                     options['atlas'][0],
                     ets_auc_denoised,
                     idx_u,
                     idx_v,
                     TR,
                     OUT_DIR,
                     history_str)
    elif options['workflow'][0] == 'pfm':
        connPFM(options['data'][0],
            options['atlas'][0],
            options['auc'][0],
            options['tr'][0],
            options['username'][0],
            options['te'],
            options['dir'],
            options['block'],
            options['jobs'][0],
            options['nsurrogates'][0],
            options['nstability'],
            options['percentile'],
            options['maxiterfactor'],
            options['hrf_shape'],
            options['hrf_path'],
            history_str)
    elif options['workflow'][0] == 'ev':
        ev.ev_workflow(options['data'][0],
            options['auc'][0],
            options['atlas'][0],
            options['dir'],
            os.path.dirname(options['auc'][0]),
            history_str)
    elif options['workflow'][0] == 'debias':
        ets_auc_denoised = loadtxt(options['matrix'][0])
        ev.debiasing(options['data'][0], 
                     options['atlas'][0],
                     ets_auc_denoised,
                     options['tr'][0],
                     os.path.dirname(options['auc'][0]),
                     history_str)
    else:
        LGR.warning(f'selected workflow {options["workflow"][0]} is not valid please reveiw possible options')


if __name__ == "__main__":
    _main(sys.argv[1:])
