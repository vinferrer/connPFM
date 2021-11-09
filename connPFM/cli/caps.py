import argparse


def _get_parser():
    parser = argparse.ArgumentParser(
        description="Calculate CAPs on activity-inducing signal calculated with connPFM."
    )

    reqoptions = parser.add_argument_group("Required options")
    reqoptions.add_argument("-i", "--input", dest="data_path", required=True, help="Input file")
    reqoptions.add_argument("-o", "--output", dest="output", required=True, help="Output file")
    reqoptions.add_argument("-a", "--atlas", dest="atlas", required=True, help="Atlas file")
    reqoptions.add_argument("-m", "--mask", dest="mask", required=True, help="ROI mask")

    optoptions = parser.add_argument_group("Optional options")
    optoptions.add_argument(
        "-sp",
        "--surrprefix",
        dest="surrprefix",
        type=str,
        default=None,
        help="surrogate prefix surrogates",
    )
    optoptions.add_argument(
        "-n",
        "--nsurrogates",
        dest="nsur",
        type=int,
        default=1000,
        help="Number of surrogates",
    )
    optoptions.add_argument(
        "-idx",
        "--idx_peak",
        dest="idx_out",
        type=str,
        default=None,
        help="selected peak timepoins file",
    )
    optoptions.add_argument(
        "-j", "--jobs", dest="njobs", type=int, default=-1, help="Number of jobs"
    )

    return parser
