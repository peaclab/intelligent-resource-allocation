#!/bin/bash -l

#$ -P peaclab-mon
#$ -N eagle_3fs_exp
#$ -o ./eagle_3fs_exp.out
#$ -l mem_per_core=18G 
#$ -pe omp 28
#$ -m ea

module load python3/3.8.10
source /project/peaclab-mon/monitoring_venv.sh
python /projectnb/peaclab-mon/boztop/resource-allocation/python_scripts/nrel_eagle_exp.py