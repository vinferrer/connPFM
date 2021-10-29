import argparse


def _get_parser():
    parser = argparse.ArgumentParser(
        description="Calculate CAPs on activity-inducing signal calculated with connPFM."
    )

    reqoptions = parser.add_argument_group("Required options")
    reqoptions.add_argument("-i", "--input", dest="data", required=True, help="Input file")
    reqoptions.add_argument("-o", "--output", dest="output", required=True, help="Output file")
    reqoptions.add_argument("-a", "--atlas", dest="atlas", required=True, help="Atlas file")

    optoptions = parser.add_argument_group("Optional options")
    optoptions.add_argument(
        "-n",
        "--nsurrogates",
        dest="nsurrogates",
        type=int,
        default=1000,
        help="Number of surrogates",
    )
    optoptions.add_argument(
        "-j", "--jobs", dest="jobs", type=int, default=1, help="Number of jobs"
    )

    return parser
