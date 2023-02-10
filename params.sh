#!/bin/bash

export MODEL_TYPE='_' # name for model
export EPOCHS=100 #how many epochs will be created
export FIG_FREQ=20 # how often figures will get generated

export SAVE_MODEL=False # true if model saves
export SAVE_FIGS=True # true if model saves figures
export RECORD=False
export CHECK_CONSISTENCY=false

#########ENCODER#########
export FULLY_CONNECTED_ENCODER=true
export TF_GROUPED_FC_INDEP_ENCODER=false
export SHALLOW_ENCODER=false

#########DECODER#########
export FULLY_CONNECTED_DECODER=true
export GENE_GROUPED_FC_INDEP_DECODER=false
export SHALLOW_DECODER=false


export SPLITS=5 # how many splits you want for cross validation
export CYCLES=2

export EN_L2=0 # L2 norm for Encoder
export DE_L2=0 # L2 norm for Decoder
export DROPOUT=0 # dropout rate
export NOISE=0 #Add Gaussian Noise to input
export BATCH_NORM=False # true if want batch norm

export MOA_BETA=0.9
export MOA_SUBSET=0
export MOA=1 # constant for MOA

export WARM_RESTART=0 # how many times you want to restart settings

export BATCH_SIZE=128 # number of training examples used in one iteration
export WIDTH_MULTIPLIER=1 #input_size*width_multiplier for width inner network
export RELATIONSHIPS_FILTER=10

export ENCODER_DEPTH=2
export DECODER_DEPTH=2

export EN_LEARNING_RATE=1e-4 # learning rate
export DE_LEARNING_RATE=1e-4 # learning rate

export EN_LR_SCHED=false # true if want learning rate to be scheduled
export EN_MAX_LR_SCHED=1e-1 # max learning rate if scheduling

export DE_LR_SCHED=false # true if want learning rate to be scheduled
export DE_MAX_LR_SCHED=1e-1 # max learning rate if scheduling

export SPARSE_DATA_PATH=/nobackup/users/schaferd/ae_project_data/dorothea_tf_gene_relationship_knowledge/dorotheaSelectionAB_1.tsv
#export SPARSE_DATA_PATH=/nobackup/users/schaferd/ae_project_data/gene_set_enrichment_analysis/gene_set.tsv
INPUT_DATA_TYPE="hdf_agg_data" #poss types: roc_agg_data, hdf_agg_data, pos_df, neg_df, pos_neg_df, blood_data
export INPUT_DATA=/home/schaferd/ae_project/input_data_processing/${INPUT_DATA_TYPE}.pkl
export TF_RNASEQ_DATA=/home/schaferd/ae_project/input_data_processing/tf_agg_data.pkl
#export ROC_DATA_PATH="/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/contrasts/"
export ROC_DATA_PATH="/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/pos_neg_samples/"
export BLOOD_META_DATA="/nobackup/users/schaferd/blood_analysis_data/SCP43/metadata/metadata.txt"
export BLOOD_DATA="/nobackup/users/schaferd/blood_analysis_data/SCP43/expression/expression_matrix_tpm.txt"

export curr_path=$(pwd)

export shallow=$curr_path/shallow/
export fc="${curr_path}/fc/"
export tf_grouped_fc_indep="${curr_path}/tf_grouped_fc_indep/"
export tf_grouped_indep="${curr_path}/tf_grouped_indep/"
export gene_grouped_fc_indep="${curr_path}/gene_grouped_fc_indep/"
export gene_grouped_indep="${curr_path}/gene_grouped_indep/"
export random_indep="${curr_path}/random_indep/"

export SAVE_PATH=/nobackup/users/schaferd/ae_project_outputs/model_eval/
export RECORD_PATH=$SAVE_PATH/model_eval.pkl

export out_path=/nobackup/users/schaferd/ae_project_outputs/model_eval/
