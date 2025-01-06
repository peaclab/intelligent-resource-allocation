#!/bin/bash -l

#$ -P peaclab-mon
#$ -N m100_3fs_new_exp
#$ -o ./m100_3fs_new_exp.out
#$ -pe omp 28
#$ -l mem_per_core=18G 
#$ -m ea

module load python3/3.8.10
source /project/peaclab-mon/monitoring_venv.sh
python /projectnb/peaclab-mon/boztop/resource-allocation/python_scripts/m100_exp.py