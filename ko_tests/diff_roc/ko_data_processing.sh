#!/bin/bash 
#SBATCH -n 1
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=30:00

module load cuda/10.2
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=ae_train
CONDA_ROOT=$HOME2/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
MODEL_TYPE=$1

python3 ko_data_processing.py
