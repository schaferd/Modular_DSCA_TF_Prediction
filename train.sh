#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o /nobackup/users/schaferd/ae_project_outputs/slurm_outputs/modular_out/slurm%j.out
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00

source params.sh

HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=ae_train3
CONDA_ROOT=$HOME2/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
#ulimit -s unlimited
module load cuda/10.2

cwd=$(pwd)



if [ "$TF_GROUPED_FC_INDEP_ENCODER" = true ] ; then
	export encoder_path=$tf_grouped_fc_indep/
	echo encoder_path $encoder_path
fi
if [ "$TF_GROUPED_FC_INDEP_DECODER" = true ] ; then
	export decoder_path=$tf_grouped_fc_indep/
	echo decoder_path $decoder_path
fi

if [ "$TF_GROUPED_INDEP_ENCODER" = true ] ; then
	export encoder_path=$tf_grouped_indep/
	echo encoder_path $encoder_path
fi
if [ "$TF_GROUPED_INDEP_DECODER" = true ] ; then
	export decoder_path=$tf_grouped_indep/
	echo decoder_path $decoder_path
fi

if [ "$GENE_GROUPED_FC_INDEP_ENCODER" = true ] ; then
	export encoder_path=$gene_grouped_fc_indep/
	echo encoder_path $encoder_path
fi
if [ "$GENE_GROUPED_FC_INDEP_DECODER" = true ] ; then
	export decoder_path=$gene_grouped_fc_indep/
	echo decoder_path $decoder_path
fi

if [ "$GENE_GROUPED_INDEP_ENCODER" = true ] ; then
	export encoder_path=$gene_grouped_indep/
	echo encoder_path $encoder_path
fi
if [ "$GENE_GROUPED_INDEP_DECODER" = true ] ;
then
	export decoder_path=$gene_grouped_indep/
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

if [ "$RANDOM_INDEP_ENCODER" = true ] ;
then
	export encoder_path=$random_indep/
	echo encoder_path $encoder_path
fi
if [ "$RANDOM_INDEP_DECODER" = true ] ;
then
	export decoder_path=$random_indep/
	echo decoder_path $decoder_path
fi
#python3 $curr_path/train.py --epochs $EPOCHS --save_path $SAVE_PATH --save_figs $SAVE_FIGS --learning_rate $LEARNING_RATE --fig_freq $FIG_FREQ --model_type $MODEL_TYPE --l2 $L2 --dropout $DROPOUT --save_model $SAVE_MODEL --sparse_data_path=$SPARSE_DATA_PATH --input_data_path=$INPUT_DATA --batch_size $BATCH_SIZE --lr_sched $LR_SCHED --batch_norm $BATCH_NORM --moa $MOA --warm_restart $WARM_RESTART --max_lr $MAX_LR_SCHED --k_splits $SPLITS --width_multiplier $WIDTH_MULTIPLIER --relationships_filter $RELATIONSHIPS_FILTER --moa_beta $MOA_BETA --moa_subset $MOA_SUBSET --roc_data_path $ROC_DATA_PATH --record $RECORD --record_path $RECORD_PATH --cycles $CYCLES --rnaseq_tf_eval_path $TF_RNASEQ_DATA --encoder_depth $ENCODER_DEPTH --decoder_depth $DECODER_DEPTH --blood_data $BLOOD_DATA --blood_meta_data $BLOOD_META_DATA 
python3 $curr_path/train.py --epochs $EPOCHS --save_path $SAVE_PATH --save_figs $SAVE_FIGS --en_learning_rate $EN_LEARNING_RATE --de_learning_rate $DE_LEARNING_RATE --fig_freq $FIG_FREQ --model_type $MODEL_TYPE --en_l2 $EN_L2 --de_l2 $DE_L2 --dropout $DROPOUT --save_model $SAVE_MODEL --sparse_data_path=$SPARSE_DATA_PATH --input_data_path=$INPUT_DATA --batch_size $BATCH_SIZE --de_lr_sched $DE_LR_SCHED --en_lr_sched $EN_LR_SCHED --batch_norm $BATCH_NORM --moa $MOA --warm_restart $WARM_RESTART --de_max_lr $DE_MAX_LR_SCHED --en_max_lr $EN_MAX_LR_SCHED --k_splits $SPLITS --width_multiplier $WIDTH_MULTIPLIER --relationships_filter $RELATIONSHIPS_FILTER --moa_beta $MOA_BETA --moa_subset $MOA_SUBSET --roc_data_path $ROC_DATA_PATH --record $RECORD --record_path $RECORD_PATH --cycles $CYCLES --rnaseq_tf_eval_path $TF_RNASEQ_DATA --encoder_depth $ENCODER_DEPTH --decoder_depth $DECODER_DEPTH --blood_data $BLOOD_DATA --blood_meta_data $BLOOD_META_DATA --noise $NOISE




