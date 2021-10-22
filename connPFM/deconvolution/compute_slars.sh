#!/bin/bash
#$ -cwd
#$ -o /bcbl/home/public/PARK_VFERRER/PFM_data/slars_out.txt
#$ -e /bcbl/home/public/PARK_VFERRER/PFM_data/slars_err.txt
#$ -S /bin/bash
#$ -q short.q

module unload python/python3.6
module load python/venv
source activate /bcbl/home/public/PARK_VFERRER/py38

python -u /bcbl/home/public/PARK_VFERRER/PFM/Scripts/compute_slars.py $INPUT_ARGS
