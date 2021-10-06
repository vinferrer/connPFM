#!/bin/bash
#$ -cwd
#$ -o /bcbl/home/public/PARK_VFERRER/PFM_data/slars_out.txt
#$ -e /bcbl/home/public/PARK_VFERRER/PFM_data/slars_err.txt
#$ -S /bin/bash
#$ -q short.q

module load python/python3.6

python -u /bcbl/home/public/PARK_VFERRER/PFM/Scripts/compute_slars.py $INPUT_ARGS
