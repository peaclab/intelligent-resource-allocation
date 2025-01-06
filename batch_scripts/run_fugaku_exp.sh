#!/bin/bash -l

#$ -P peaclab-mon
#$ -N fugaku_3fs_exp_new
#$ -o ./fugaku_3fs_exp_new.out
#$ -l mem_per_core=18G 
#$ -pe omp 28
#$ -m ea

module load python3/3.8.10
source /project/peaclab-mon/monitoring_venv.sh
python /projectnb/peaclab-mon/boztop/resource-allocation/python_scripts/fugaku_exp.py