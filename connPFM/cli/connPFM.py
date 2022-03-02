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
        "-i", "--input", dest="data", required=True, help="Input data.", nargs="+", type=str
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
        type=float,
        default=None,
        help=(
            "List of echo times (default=None). If no TE is given, "
            "the single echo version will be run."
        ),
        nargs="+",
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
        default="all",
        choices=["pfm", "ev", "debias", "all"],
        help=(
            "Possiblility of executing different parts of the workflow:"
            "pfm: calculates only the AUC dataset"
            "ev: calculates the ets matrix (--AUC dataset required as argument) "
            "debias: Calculates fitted dataset based on ets matrix (--matrix required as argument)"
            "all: executes all the workflow"
        ),
        nargs=1,
    )
    optoptions.add_argument(
        "-m",
        "--matrix",
        dest="matrix",
        help="Path to the connectivity matrix.",
        default=None,
        type=str,
        nargs=1,
    )
    optoptions.add_argument(
        "-peaks",
        "--peaks",
        dest="peak_detection",
        type=str,
        default=["rss"],
        choices=["rss", "rss_time", "ets", "ets_time"],
        help=(
            "Method to detect peaks of co-fluctuations:"
            "- rss: significant peaks in root sum of squares."
            "- rss_time: significant peaks in root sum of squares, but p-values are"
            "based on each time-point."
            "- ets: significant peaks in edge-time matrix."
            "- ets_time: significant peaks in edge-time matrix, but thresholds are"
            "based on each time-point."
        ),
        nargs=1,
    )
    optoptions.add_argument(
        "-pd",
        "--prefix_debias",
        dest="prefix",
        help="Prefix for path and name for the beta and fitted files of the debiasing",
        default=None,
    )
    optoptions.add_argument(
        "-pp",
        "--peaks_points",
        dest="peaks_path",
        help="Prefix for name for txt file of rss and txt file of selected timepoints",
        default=None,
    )
    optoptions.add_argument(
        "-groups",
        "--groups",
        dest="groups",
        help="Whether to group continuous spikes when debiasing.",
        default=False,
        action="store_true",
    )
    optoptions.add_argument(
        "-dist",
        "--distance",
        dest="groups_dist",
        help="Distance (TRs) between groups when grouping continuous spikes.",
        default=3,
        type=int,
        nargs=1,
    )
    optoptions.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        action="store_true",
        help="activate quiet logger mode",
        default=False,
    )
    optoptions.add_argument(
        "-dg",
        "--debug",
        dest="debug",
        action="store_true",
        help="activate quiet logger mode",
        default=False,
    )

    return parser
