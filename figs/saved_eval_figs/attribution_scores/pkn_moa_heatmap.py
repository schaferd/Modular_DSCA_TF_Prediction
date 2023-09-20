import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)

base_path = '/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/'
fc_g_path = [base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.56/',base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/']
input_genes = pd.read_pickle(base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/fold0_cycle0/ko_activities_cycle0_fold0/input_genes.pkl')

pknAB = pd.read_csv('pknAB.csv',sep='\t')
pknCD = pd.read_csv('pknCD.csv',sep='\t')

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
"""
attr_dicts = [pd.read_pickle(f) for f in fc_g_attr_files]
"""

#FOR JUST ONE RUN
attr_dicts = [pd.read_pickle(f) for f in [fc_g_attr_files[0]]]

tf_attr_dict = {}
for model_attr in attr_dicts:
    for tf in model_attr.keys():
        if tf not in tf_attr_dict:
            #FOR MULTIPLE RUNS
            #tf_attr_dict[tf] = [np.array([model_attr[tf]])]

            #FOR JUST ONE RUN
            tf_attr_dict[tf] = np.array(model_attr[tf])
        else:
            print("hello")
            tf_attr_dict[tf].append(np.array([model_attr[tf]]))

"""
#tf_attr_dict = {tf:np.vstack(tf_attr_dict[tf]) for tf in tf_attr_dict.keys()}
#tf_attr_mean_dict = {tf:np.mean(np.mean(tf_attr_dict[tf],axis=0),axis=0) for tf in tf_attr_dict.keys()}
attr_df = pd.DataFrame(tf_attr_mean_dict,index=input_genes)
"""

#FOR JUST ONE RUN
tf_attr_mean_dict = {tf:np.mean(tf_attr_dict[tf],axis=0) for tf in tf_attr_dict.keys()}
attr_df = pd.DataFrame(tf_attr_mean_dict,index=input_genes)
###

gene_to_ensembl = {}
ensembl_to_gene = {}

for g in attr_df.index:
        try:
                gene_id = ensembl_data.gene_name_of_gene_id(g)
                gene_to_ensembl[gene_id] = g 
                ensembl_to_gene[g] = gene_id
        except:
                ensembl_to_gene[g] = None

attr_df = attr_df.rename(index=ensembl_to_gene).T

pkn_moa_df = pd.DataFrame(np.zeros(attr_df.shape),index=attr_df.index,columns=attr_df.columns)
print(pkn_moa_df.shape)
input_genes_set = set([ensembl_to_gene[g] for g in input_genes])
tf_set = set(attr_df.index)

for i,row in pknAB.iterrows():
    gene = row['target']
    tf = row['source']
    moa = 1 if row['mor'] > 0 else -1
    if gene in input_genes_set and tf in tf_set:
        pkn_moa_df.at[tf,gene] = moa 

pd.to_pickle(pkn_moa_df,"pkn_moa_df.pkl")

plt.clf()
fig,ax = plt.subplots()
fig.set_figwidth(18)
fig.set_figheight(15)
clustergrid = sns.clustermap(pkn_moa_df,center=0,figsize=(14,12),cmap='vlag',vmin=-1,vmax=1)
plt.savefig('pkn_heatmap.png')

pd.to_pickle(pkn_moa_df.index[clustergrid.dendrogram_row.reordered_ind],'pkn_row_indices.pkl')
pd.to_pickle(pkn_moa_df.columns[clustergrid.dendrogram_col.reordered_ind],'pkn_col_indices.pkl')



