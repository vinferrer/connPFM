#!/usr/bin/python
import argparse
import logging
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

from connPFM.deconvolution.stability_lars import StabilityLars

LGR = logging.getLogger(__name__)


def main(argv):

    parser = argparse.ArgumentParser(description="Stability Selection with LARS")
    # parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
    # help='Shows the list of possible commands and a brief description of each.')
    parser.add_argument("--data", type=str, help="Data to analyse.", nargs=1)
    parser.add_argument("--hrf", type=str, help="HRF matrix (design matrix).", nargs=1)
    parser.add_argument("--voxel", type=int, help="Voxel number to analyse.", default=1, nargs=1)
    parser.add_argument("--nscans", type=int, help="Number of scans in data.", nargs=1)
    parser.add_argument(
        "--maxiterfactor",
        type=float,
        help=(
            "Maximum number of iterations in the LARS procedure is relative to "
            "the number of volumes of the input time series, i.e.  maxiterfactor * nscans"
        ),
        nargs=1,
    )
    parser.add_argument(
        "--nsurrogates",
        type=int,
        help="Number of surrogates to use (default=100).",
        default=100,
        nargs=1,
    )
    parser.add_argument(
        "--nte",
        type=int,
        help="Number of TE in the signal to analyse (default=1).",
        default=1,
        nargs=1,
    )
    parser.add_argument(
        "--mode",
        type=int,
        help=(
            "Subsampling mode: 1 = different time points are selected across echoes. 2 = same "
            "time points are selected across echoes. (default=1)."
        ),
        default=1,
        nargs=1,
    )
    parser.add_argument(
        "--tempdir",
        type=str,
        help='Temporary directory to store temporary files (default="/temp").',
        default=None,
        nargs=1,
    )
    parser.add_argument("--first", type=int, help="First voxel idx.", default=None, nargs=1)
    parser.add_argument("--last", type=int, help="Last voxel idx.", default=None, nargs=1)
    parser.add_argument("--voxels", type=int, help="Total amount of voxels.", default=1, nargs=1)
    parser.add_argument("--n_job", type=int, help="Job number.", nargs=1)
    args = parser.parse_args()

    LGR.info("Data file is ", args.data[0])
    LGR.info("HRF file is ", args.hrf[0])
    LGR.info("Voxel is: ", args.voxel)
    LGR.info("nscans is: ", args.nscans[0])
    LGR.info("N surrogates: ", args.nsurrogates[0])
    LGR.info("N TE: ", args.nte[0])
    LGR.info("Mode: ", args.mode[0])
    LGR.info("Dir: ", args.tempdir[0])

    sl = StabilityLars()

    if type(args.data) is list:
        data_filename = args.data[0]
    else:
        data_filename = args.data
    if type(args.hrf) is list:
        hrf_file = args.hrf[0]
    else:
        hrf_file = args.hrf
    if type(args.voxel) is list:
        voxel = int(args.voxel[0])
    else:
        voxel = int(args.voxel)
    if type(args.nscans) is list:
        nscans = int(args.nscans[0])
    else:
        nscans = int(args.nscans)
    if type(args.maxiterfactor) is list:
        maxiterfactor = int(args.maxiterfactor[0])
    else:
        maxiterfactor = int(args.maxiterfactor)
    if type(args.nsurrogates) is list:
        sl.nsurrogates = int(args.nsurrogates[0])
    else:
        sl.nsurrogates = int(args.nsurrogates)
    if type(args.nte) is list:
        sl.nTE = int(args.nte[0])
    else:
        sl.nTE = int(args.nte)
    if type(args.mode) is list:
        sl.mode = int(args.mode[0])
    else:
        sl.mode = int(args.mode)
    # if type(args.tempdir) is list: seems like tempdir is reassined later
    #     tempdir = args.tempdir[0]
    # else:
    #     tempdir = args.tempdir
    if type(args.first) is list:
        first = args.first[0]
    else:
        first = args.first
    if type(args.last) is list:
        last = args.last[0]
    else:
        last = args.last
    if type(args.voxels) is list:
        voxels_total = args.voxels[0]
    else:
        voxels_total = args.voxels
    if type(args.n_job) is list:
        n_job = args.n_job[0]
    else:
        n_job = args.n_job

    if first is None:
        nvoxels = 1
    else:
        nvoxels = last - first + 1
        voxel = first

    if last == voxels_total:
        nvoxels = nvoxels - 1

    y = np.load(data_filename)
    auc = np.zeros((nscans, nvoxels))
    sl.key = first
    sl.maxiterfactor = maxiterfactor

    LGR.info("Job number: {}".format(n_job))
    LGR.info("First voxel: {}".format(first))
    LGR.info("Last voxel: {}".format(last))
    LGR.info("Number of voxels: {}".format(nvoxels))

    hrf = np.load(hrf_file)
    for vox_idx in range(nvoxels):
        sl.stability_lars(hrf, np.expand_dims(y[:, voxel + vox_idx], axis=-1))
        auc[:, vox_idx] = np.squeeze(sl.auc)
        LGR.info(
            "AUC of voxel {}/{} calculated and stored...".format(str(vox_idx + 1), str(nvoxels))
        )

    filename = args.tempdir[0] + "/auc_" + str(n_job) + ".npy"
    np.save(filename, auc)


if __name__ == "__main__":
    main(sys.argv[1:])
