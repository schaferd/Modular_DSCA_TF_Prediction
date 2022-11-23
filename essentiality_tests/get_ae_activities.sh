#!/bin/bash 
#SBATCH -n 1
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=30:00
#SBATCH --gres=gpu:1

module load cuda/10.2
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=ae_train3
CONDA_ROOT=$HOME2/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
MODEL_TYPE=$1
if [ "$MODEL_TYPE" = 'v' ]
then
        export MODEL_PATH=""
        export encoder_path="/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/"
elif [ "$MODEL_TYPE" = 'd' ]
then
        export MODEL_PATH=""
        export encoder_path="/home/schaferd/ae_project/attempt2/transcriptomics_autoencoder/decomp/"
elif [ "$MODEL_TYPE" = 'f' ]
then
        export MODEL_PATH=""
        export encoder_path="/home/schaferd/ae_project/curr_gpu/transcriptomics_autoencoder/for_loop/"
fi
python3 get_ae_activities.py
