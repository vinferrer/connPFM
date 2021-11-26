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
module load singularity
if ! command -v singularity &> /dev/null;
then
    echo " singularity could not be found tryng to execute with conda env"
    module unload python/python3.6
    module load python/venv
    source activate /bcbl/home/public/PARK_VFERRER/py38
    SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

    python -u $SCRIPT_DIR/compute_slars.py $INPUT_ARGS
else
    echo ${INPUT_ARGS[0]}
    cd /bcbl/home/public/PARK_VFERRER
    singularity exec --bind $HOME $HOME/connpfm_slim.simg python -u /connPFM/connPFM/deconvolution/compute_slars.py $INPUT_ARGS
fi
