#!/bin/bash
#$ -cwd
#$ -o /bcbl/home/public/PARK_VFERRER/PFM_data/slars_out.txt
#$ -e /bcbl/home/public/PARK_VFERRER/PFM_data/slars_err.txt
#$ -S /bin/bash
#$ -q short.q
if [[ -z "${INPUT_ARGS}" ]]; then
    if [[ ! -z "$1" ]]; then
        INPUT_ARGS="$*"
    fi
fi
module unload python/python3.6
module load python/venv
source activate /bcbl/home/public/PARK_VFERRER/py38
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python -u $SCRIPT_DIR/compute_slars.py $INPUT_ARGS
