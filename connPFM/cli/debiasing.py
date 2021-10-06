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
        "-auc", "--auc", dest="auc", required=True, help="AUC dataset.", nargs="+"
    )
    reqoptions.add_argument(
        "-o",
        "--outdir",
        dest="outdir",
        required=True,
        help="Name of the output directory.",
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
        "-nsurrogates",
        "--nsurrogates",
        dest="nsurrogates",
        type=int,
        help=("Number of surrogates to calculate AUC on (default = 0)."),
        default=0,
        nargs=1,
    )
    optoptions.add_argument(
        "-p",
        "--percentile",
        dest="percent",
        type=int,
        help=("Number of surrogates to calculate AUC on (default = 0)."),
        default=95,
        nargs=1,
    )
    return parser


def plotting_parser():
    """
    Parse command line inputs for aroma.

    Returns
    -------
    parser.parse_args() : argparse dict
    """
    parser = argparse.ArgumentParser(description=("Plotting ets-matrix script "))

    # Required options
    reqoptions = parser.add_argument_group("Required arguments")
    reqoptions.add_argument(
        "-d",
        "--directory",
        dest="dir",
        required=True,
        help="Input directory.",
        type=str,
        nargs="+",
    )
    reqoptions.add_argument(
        "-s", "--subject", dest="subject", required=True, help="Subject", type=str, nargs="+"
    )
    reqoptions.add_argument(
        "-nr", "--nroi", dest="nROI", required=True, help="numero de ROI", type=str, nargs="+"
    )
    return parser
