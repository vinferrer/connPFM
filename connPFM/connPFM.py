import datetime
import getpass
import logging
import os
import socket
import sys

from numpy import loadtxt

from connPFM.cli.connPFM import _get_parser
from connPFM.connectivity.ev import ev_workflow
from connPFM.debiasing.debiasing import debiasing
from connPFM.deconvolution.roiPFM import roiPFM
from connPFM.utils import loggers

LGR = logging.getLogger(__name__)
LGR.setLevel(logging.INFO)


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    options = vars(options)
    args_str = str(options)[9:]
    history_str = "[{username}@{hostname}: {date}] connPFM with {arguments}".format(
        username=getpass.getuser(),
        hostname=socket.gethostname(),
        date=datetime.datetime.now().strftime("%c"),
        arguments=args_str,
    )
    # create logfile name
    dir = os.path.abspath(options["dir"])
    os.makedirs(dir, exist_ok=True)

    LGR = logging.getLogger("GENERAL")
    basename = "connPFM_"
    extension = "tsv"
    start_time = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    logname = os.path.join(dir, (basename + start_time + "." + extension))
    refname = os.path.join(dir, "_references.txt")
    loggers.setup_loggers(logname, refname, quiet=options["quiet"], debug=options["debug"])

    if type(options["workflow"]) is list:
        selected_workflow = options["workflow"][0]
    else:
        selected_workflow = options["workflow"]

    if selected_workflow == "all":
        roiPFM(
            options["data"][0],
            options["atlas"][0],
            options["auc"][0],
            options["tr"][0],
            options["username"][0],
            options["te"],
            dir,
            options["block"],
            options["jobs"][0],
            options["nsurrogates"][0],
            options["nstability"],
            options["percentile"],
            options["maxiterfactor"],
            options["hrf_shape"],
            options["hrf_path"],
            history_str,
        )

        ets_auc_denoised = ev_workflow(
            options["data"][0],
            options["auc"][0],
            options["atlas"][0],
            dir,
            os.path.dirname(options["auc"][0]),
            options["matrix"][0],
            options["nsurrogates"][0],
            history_str,
            options["peak_detection"][0],
            afni_text=options["peaks_path"],
        )
        LGR.info("Perform debiasing based on edge-time matrix.")
        debiasing(
            options["data"][0],
            options["atlas"][0],
            ets_auc_denoised,
            options["tr"][0],
            os.path.dirname(options["data"][0]),
            options["prefix"],
            options["groups"],
            options["groups_dist"],
            history_str,
        )
    elif selected_workflow == "pfm":
        roiPFM(
            options["data"][0],
            options["atlas"][0],
            options["auc"][0],
            options["tr"][0],
            options["username"][0],
            options["te"],
            dir,
            options["block"],
            options["jobs"][0],
            options["nsurrogates"][0],
            options["nstability"],
            options["percentile"],
            options["maxiterfactor"],
            options["hrf_shape"],
            options["hrf_path"],
            history_str,
        )
    elif selected_workflow == "ev":
        ev_workflow(
            data_file=options["data"][0],
            auc_file=options["auc"][0],
            atlas=options["atlas"][0],
            surr_dir=dir,
            out_dir=os.path.dirname(options["auc"][0]),
            matrix=options["matrix"][0],
            nsurrogates=options["nsurrogates"][0],
            history_str=history_str,
            peak_detection=options["peak_detection"][0],
            afni_text=options["peaks_path"],
        )
    elif selected_workflow == "debias":
        ets_auc_denoised = loadtxt(options["matrix"][0])
        debiasing(
            options["data"][0],
            options["atlas"][0],
            ets_auc_denoised,
            options["tr"][0],
            os.path.dirname(options["data"][0]),
            options["prefix"],
            options["groups"],
            options["groups_dist"],
            history_str,
        )
    else:
        LGR.warning(
            f"selected workflow {selected_workflow} is not valid please " "review possible options"
        )
    loggers.teardown_loggers()


if __name__ == "__main__":
    _main(sys.argv[1:])
