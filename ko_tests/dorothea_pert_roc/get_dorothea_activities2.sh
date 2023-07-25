#!/bin/bash
#SBATCH --mem=64GB
#SBATCH --time=20:00:00

HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=ae_eval2
CONDA_ROOT=$HOME2/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

#Rscript get_dorothea_activities_pert.r 
Rscript get_dorothea_benchmark_activities_diff_inferred.r
