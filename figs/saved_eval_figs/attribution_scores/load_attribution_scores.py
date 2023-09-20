import pandas as pd
import numpy as np
import os
import sys
from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)

base_path = '/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/'
fc_g_path = [base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.56/',base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/']
input_genes = pd.read_pickle(base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/fold0_cycle0/ko_activities_cycle0_fold0/input_genes.pkl')

def get_attr_scores_one_run():

    fc_g_attr_files = []

    for path in fc_g_path:
        fold_paths = []
        fold_paths = fold_paths+[path+'/'+p for p in os.listdir(path) if 'fold' in p]
        act_path = []
        for p in fold_paths:
            act_path = act_path + [p+'/'+i for i in os.listdir(p) if 'ko_activities' in i]
            for p2 in act_path:
                fc_g_attr_files = fc_g_attr_files+[p2+'/'+f for f in os.listdir(p2) if 'tf_attr_dict.pkl' in f]

    #FOR JUST ONE RUN
    attr_dicts = [pd.read_pickle(f) for f in [fc_g_attr_files[0]]]

    tf_attr_dict = {}
    for model_attr in attr_dicts:
        for tf in model_attr.keys():
            if tf not in tf_attr_dict:
                #FOR JUST ONE RUN
                tf_attr_dict[tf] = np.array(model_attr[tf])

    #FOR JUST ONE RUN
    tf_attr_mean_dict = {tf:np.mean(tf_attr_dict[tf],axis=0) for tf in tf_attr_dict.keys()}
    attr_df = pd.DataFrame(tf_attr_mean_dict,index=input_genes)
    ###
    return attr_df

def ensembl_to_gene_name(attr_df,genes_in_row=True):
    gene_to_ensembl = {}
    ensembl_to_gene = {}
    genes = attr_df.columns
    if genes_in_row:
        genes = attr_df.index

    for g in genes:
            try:
                    gene_id = ensembl_data.gene_name_of_gene_id(g)
                    gene_to_ensembl[gene_id] = g 
                    ensembl_to_gene[g] = gene_id
            except:
                    ensembl_to_gene[g] = None

    if genes_in_row:
        attr_df = attr_df.rename(index=ensembl_to_gene)
    else:
        attr_df = attr_df.rename(columns=ensembl_to_gene)

    return attr_df


def get_attr_scores_avg_runs():

    fc_g_attr_files = []

    for path in fc_g_path:
        fold_paths = []
        fold_paths = fold_paths+[path+'/'+p for p in os.listdir(path) if 'fold' in p]
        act_path = []
        for p in fold_paths:
            act_path = act_path + [p+'/'+i for i in os.listdir(p) if 'ko_activities' in i]
            for p2 in act_path:
                fc_g_attr_files = fc_g_attr_files+[p2+'/'+f for f in os.listdir(p2) if 'tf_attr_dict.pkl' in f]

    #FOR MULTIPLE RUNS
    attr_dicts = [pd.read_pickle(f) for f in fc_g_attr_files]

    tf_attr_dict = {}
    for model_attr in attr_dicts:
        for tf in model_attr.keys():
            if tf not in tf_attr_dict:
                #FOR MULTIPLE RUNS
                tf_attr_dict[tf] = [np.array([model_attr[tf]])]

            else:
                tf_attr_dict[tf].append(np.array([model_attr[tf]]))

    tf_attr_dict = {tf:np.vstack(tf_attr_dict[tf]) for tf in tf_attr_dict.keys()}
    tf_attr_mean_dict = {tf:np.mean(np.mean(tf_attr_dict[tf],axis=0),axis=0) for tf in tf_attr_dict.keys()}
    attr_df = pd.DataFrame(tf_attr_mean_dict,index=input_genes)

    return attr_df
