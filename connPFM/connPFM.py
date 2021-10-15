import datetime
import getpass
import logging
import os
import socket
import sys

from cli.connPFM import _get_parser
from connectivity import ev
from deconvolution.roiPFM import roiPFM
from numpy import loadtxt
from utils import loggers

LGR = logging.getLogger(__name__)
LGR.setLevel(logging.INFO)


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
    # create logfile name
    dir = os.path.abspath(options["dir"])
    os.makedirs(dir, exist_ok=True)

    LGR = logging.getLogger("GENERAL")
    basename = "connPFM_"
    extension = "tsv"
    start_time = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    logname = os.path.join(dir, (basename + start_time + "." + extension))
    refname = os.path.join(dir, "_references.txt")
    loggers.setup_loggers(logname, refname,
                          quiet=options['quiet'],
                          debug=options['debug']
                          )
    if options["workflow"][0] == "all":
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

        ets_auc_denoised = ev.ev_workflow(
            options["data"][0],
            options["auc"][0],
            options["atlas"][0],
            options["nsurrogates"],
            os.path.abspath(options["auc"][0]),
            history_str,
        )
        LGR.info("Perform debiasing based on edge-time matrix.")
        ev.debiasing(
            options["data"][0],
            options["atlas"][0],
            ets_auc_denoised,
            options["tr"][0],
            os.path.dirname(options["auc"][0]),
            history_str,
        )
    elif options["workflow"][0] == "pfm":
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
    elif options["workflow"][0] == "ev":
        ev.ev_workflow(
            options["data"][0],
            options["auc"][0],
            options["atlas"][0],
            dir,
            os.path.dirname(options["auc"][0]),
            history_str,
        )
    elif options["workflow"][0] == "debias":
        ets_auc_denoised = loadtxt(options["matrix"][0])
        ev.debiasing(
            options["data"][0],
            options["atlas"][0],
            ets_auc_denoised,
            options["tr"][0],
            os.path.dirname(options["auc"][0]),
            history_str,
        )
    else:
        LGR.warning(
            f'selected workflow {options["workflow"][0]} is not valid please '
            'review possible options'
        )
    loggers.teardown_loggers()


if __name__ == "__main__":
    _main(sys.argv[1:])
