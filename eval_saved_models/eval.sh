#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o /home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/slurm/slurm%j.out
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00

source params.sh

HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=ae_train
CONDA_ROOT=$HOME2/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
#ulimit -s unlimited
module load cuda/10.2


if [ "$TF_GROUPED_FC_INDEP_ENCODER" = true ] ; then
	export encoder_path=$tf_grouped_fc_indep/
	echo encoder_path $encoder_path
fi


if [ "$GENE_GROUPED_FC_INDEP_DECODER" = true ] ; then
	export decoder_path=$gene_grouped_fc_indep/
	echo decoder_path $decoder_path
fi


if [ "$SHALLOW_ENCODER" = true ] ;
then
	export encoder_path=$shallow/
	echo encoder_path $encoder_path
fi
if [ "$SHALLOW_DECODER" = true ] ;
then
	export decoder_path=$shallow/
	echo decoder_path $decoder_path
fi

if [ "$FULLY_CONNECTED_ENCODER" = true ] ;
then
	export encoder_path=$fc/
	echo encoder_path $encoder_path
fi
if [ "$FULLY_CONNECTED_DECODER" = true ] ;
then
	export decoder_path=$fc/
	echo decoder_path $decoder_path
fi


python3 eval.py --model_path  $MODEL_PATH --encoder_depth $ENCODER_DEPTH --decoder_depth $DECODER_DEPTH --train_data $TRAIN_DATA --width_multiplier $WIDTH_MULTIPLIER --prior_knowledge $PRIOR_KNOWLEDGE_PATH --relationships_filter $RELATIONSHIPS_FILTER
