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
        "--output",
        dest="output",
        required=True,
        help="Name of the output dataset.",
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
        "-nsurrogates",
        "--nsurrogates",
        dest="nsurrogates",
        type=int,
        help=("Number of surrogates to calculate AUC on (default = 0)."),
        default=0,
        nargs=1,
    )
    return parser
