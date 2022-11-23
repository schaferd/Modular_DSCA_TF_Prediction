#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=128GB


HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=ae_train3
CONDA_ROOT=$HOME2/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
#ulimit -s unlimited


python3 gene_overlap_fig.py
