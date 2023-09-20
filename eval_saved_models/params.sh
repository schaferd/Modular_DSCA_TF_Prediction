

#########ENCODER#########
export FULLY_CONNECTED_ENCODER=false
export TF_GROUPED_FC_INDEP_ENCODER=false
export SHALLOW_ENCODER=true

#########DECODER#########
export FULLY_CONNECTED_DECODER=false
export GENE_GROUPED_FC_INDEP_DECODER=false
export SHALLOW_DECODER=true

export ENCODER_DEPTH=2
export DECODER_DEPTH=2
export WIDTH_MULTIPLIER=1
export RELATIONSHIPS_FILTER=10

export MODEL_PATH=/nobackup/users/schaferd/ae_project_outputs/final_eval/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/fold0_cycle0/model_encoder_cycle0_fold0.pth

#FC-G
export RUN_PATH=/nobackup/users/schaferd/ae_project_outputs/final_eval/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.56/

#S-S
#export RUN_PATH=/nobackup/users/schaferd/ae_project_outputs/final_eval/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/


export PRIOR_KNOWLEDGE_PATH=/nobackup/users/schaferd/ae_project_data/dorothea_tf_gene_relationship_knowledge/dorotheaSelectionAB_1.tsv
INPUT_DATA_TYPE="hdf_agg_data" #poss types: roc_agg_data, hdf_agg_data, pos_df, neg_df, pos_neg_df, blood_data
export TRAIN_DATA=/home/schaferd/ae_project/input_data_processing/${INPUT_DATA_TYPE}.pkl

export prev_path=$(pwd)/../
echo $prev_path
export shallow=$prev_path/shallow/
export fc="${prev_path}/fc/"
export tf_grouped_fc_indep="${prev_path}/tf_grouped_fc_indep/"
export gene_grouped_fc_indep="${prev_path}/gene_grouped_fc_indep/"
