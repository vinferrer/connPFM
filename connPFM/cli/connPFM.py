import argparse


def _get_parser():
    """
    Parse command line inputs for aroma.

    Returns
    -------
    parser.parse_args() : argparse dict
    """
    parser = argparse.ArgumentParser(
        description=(
            "Script to calculate the probability of a neuronal "
            "event ocurring at a given time-point with Paradigm "
            "Free Mapping."
        )
    )

    # Required options
    reqoptions = parser.add_argument_group("Required arguments")
    reqoptions.add_argument(
        "-i", "--input", dest="data", required=True, help="Input dataset.", nargs="+"
    )
    reqoptions.add_argument(
        "-a", "--atlas", dest="atlas", required=True, help="Input ROI or Atlas.", type=str, nargs=1
    )
    reqoptions.add_argument(
        "-o",
        "--AUC",
        dest="auc",
        required=True,
        help="Name of the auc dataset.",
        type=str,
        nargs=1,
    )
    reqoptions.add_argument(
        "-tr",
        "--tr",
        dest="tr",
        type=float,
        required=True,
        help=("TR of the acquisition."),
        nargs=1,
    )
    reqoptions.add_argument(
        "-u",
        "--username",
        dest="username",
        required=True,
        help="Username for HPC cluster.",
        type=str,
        nargs=1,
    )

    # Optional options
    optoptions = parser.add_argument_group("Optional arguments")
    optoptions.add_argument(
        "-te",
        "--te",
        dest="te",
        type=list,
        default=None,
        help=(
            "List of echo times (default=None). If no TE is given, "
            "the single echo version will be run."
        ),
        nargs=1,
    )
    optoptions.add_argument(
        "-d",
        "--dir",
        dest="dir",
        type=str,
        default="temp",
        help=("Temporary directory to store computed files (default='temp')."),
    )
    optoptions.add_argument(
        "-block",
        "--block",
        dest="block",
        action="store_true",
        help="Whether the AUC was calculated with the block model formulation (default=False).",
        default=False,
    )
    optoptions.add_argument(
        "-jobs",
        "--jobs",
        dest="jobs",
        type=int,
        help=("Number of jobs to run in parallel (default = 50)."),
        default=50,
        nargs=1,
    )
    optoptions.add_argument(
        "-nsur",
        "--nsurrogates",
        dest="nsurrogates",
        type=int,
        help=("Number of surrogates to calculate AUC on (default = 50)."),
        default=50,
        nargs=1,
    )
    optoptions.add_argument(
        "-p",
        "--percentile",
        dest="percentile",
        type=int,
        help=("Percentile used to threshold edge-time matrix (default = 95)."),
        default=95,
        nargs=1,
    )
    optoptions.add_argument(
        "-nstability",
        "--nstability",
        dest="nstability",
        type=int,
        help=(
            "Number of stability-selection surrogates to calculate probability of coefficients "
            "(default = 50)."
        ),
        default=50,
        nargs=1,
    )
    optoptions.add_argument(
        "-max",
        "--maxiterfactor",
        dest="maxiterfactor",
        type=float,
        help=(
            "Factor that multiplies the number of TRs to set the maximum number of iterations for "
            "LARS (default = 0.3)."
        ),
        default=0.3,
        nargs=1,
    )
    optoptions.add_argument(
        "-hrf",
        "--hrf",
        dest="hrf_shape",
        type=str,
        help=("HRF shape to generate with 3dDeconvolve (default = SPMG1)."),
        default="SPMG1",
        nargs=1,
    )
    optoptions.add_argument(
        "-custom_hrf",
        "--custom_hrf",
        dest="hrf_path",
        type=str,
        help=("TXT or 1D file containing an HRF (default = None)."),
        default=None,
        nargs=1,
    )
    optoptions.add_argument(
        "-w",
        "--workflow",
        dest="workflow",
        type=str,
        default=["all"],
        help=(
            "Possiblility of executing different parts of the workflow:"
            "pfm: calculates only the AUC dataset"
            "ev: calculates the ets matrix (--AUC dataset required as argument) "
            "debias: Calculates fitted dataset based on ets matrix (--matrix required as argument)"
            "all: executes all the workflow"
        ),
        nargs=1,
    )
    reqoptions.add_argument(
        "-m",
        "--matrix",
        dest="matrix",
        help="Name of the auc dataset.",
        default=None,
        type=str,
        nargs=1,
    )
    return parser
