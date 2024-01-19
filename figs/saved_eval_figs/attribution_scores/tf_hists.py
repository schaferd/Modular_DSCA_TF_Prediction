import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy import stats
import statsmodels.api as sm
from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)
from load_attribution_scores import *

base_path = '/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/'
fc_g_path = [base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.56/',base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/']
input_genes = pd.read_pickle(base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/fold0_cycle0/ko_activities_cycle0_fold0/input_genes.pkl')

pknAB = pd.read_csv('pknAB.csv',sep='\t')
pknCD = pd.read_csv('pknCD.csv',sep='\t')
print(pknAB)

"""
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

#FOR JUST ONE RUN
attr_dicts = [pd.read_pickle(f) for f in [fc_g_attr_files[0]]]

tf_attr_dict = {}
for model_attr in attr_dicts:
    for tf in model_attr.keys():
        if tf not in tf_attr_dict:
            #FOR MULTIPLE RUNS
            tf_attr_dict[tf] = [np.array([model_attr[tf]])]

            #FOR JUST ONE RUN
            #tf_attr_dict[tf] = np.array(model_attr[tf])
        else:
            print("hello")
            tf_attr_dict[tf].append(np.array([model_attr[tf]]))

#tf_attr_dict = {tf:np.vstack(tf_attr_dict[tf]) for tf in tf_attr_dict.keys()}
#tf_attr_mean_dict = {tf:np.mean(np.mean(tf_attr_dict[tf],axis=0),axis=0) for tf in tf_attr_dict.keys()}
attr_dfs
#attr_df = pd.DataFrame(tf_attr_mean_dict,index=input_genes)

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
gene_name_input_genes = [ensembl_to_gene[g] for g in input_genes]

"""

attr_dfs = [ensembl_to_gene_name(df,genes_in_row=True) for df in get_attr_scores_all_runs()] 


"""
#find TF in AB pkn of interest (TF with the most connections)
pknAB_tfs = list(set(pknAB['source']))
tf_with_most_genes = None
no_of_genes_of_tf_with_most_genes = 0
for tf in pknAB_tfs:
    no_of_genes = len(pknAB[pknAB['source'] == tf])
    if no_of_genes > no_of_genes_of_tf_with_most_genes and tf in attr_df.columns:
        no_of_genes_of_tf_with_most_genes = no_of_genes
        tf_with_most_genes = tf

print(no_of_genes_of_tf_with_most_genes)
"""

def get_top_attr_rels(attr_df, rank=False):

    abs_attr_df = attr_df.abs()

    if rank == True:
        orig_index = abs_attr_df.index
        new_df = abs_attr_df.reset_index(drop=True)
        original_positions = new_df.reset_index().melt(id_vars='index')
        original_positions['rank'] = original_positions['value'].rank(pct=True)

        abs_attr_df = original_positions.pivot(index='index',columns='variable',values='rank')
        abs_attr_df.index = orig_index

    ABscores = []
    CDscores = []
    other_scores = []


    for tf in attr_df.columns:
        print(tf)
        #Get AB and CD PKN TF relationship genes
        ABgenes = pknAB[pknAB['source'] == tf]['target']
        CDgenes = pknCD[pknCD['source'] == tf]['target']
        CDgenes = list(set(CDgenes).intersection(set(attr_df.index)))
        ABgenes = list(set(ABgenes).intersection(set(attr_df.index)))
        other = list((set(attr_df.index) - set(ABgenes)) - set(CDgenes))

        ABscores.extend(abs_attr_df.loc[ABgenes,tf])
        CDscores.extend(abs_attr_df.loc[CDgenes,tf])
        other_scores.extend(abs_attr_df.loc[other,tf])
        #print(other_scores)

    confidence_groups = {'AB':ABscores,'CD':CDscores,'Other':other_scores}
    return confidence_groups



confidence_groups = {'AB':[],'CD':[],'Other':[]}
for df in attr_dfs:
    c_group = get_top_attr_rels(df,rank=False)
    confidence_groups['AB'].extend(c_group['AB'])
    confidence_groups['CD'].extend(c_group['CD'])
    confidence_groups['Other'].extend(c_group['Other'])

confidence_df = pd.concat([pd.DataFrame({'Confidence Group':'AB','Scores':confidence_groups['AB']}),pd.DataFrame({'Confidence Group':'CD','Scores':confidence_groups['CD']}),pd.DataFrame({'Confidence Group':'Other','Scores':confidence_groups['Other']})])
confidence_df['Scores'] = pd.to_numeric(confidence_df['Scores'])

pd.to_pickle(confidence_df,"confidence_df_tf_hists.pkl")

confidence_df = pd.read_pickle("confidence_df_tf_hists.pkl")

def make_tf_hists(ax):
    kde = sns.kdeplot(data=confidence_df,x='Scores',hue='Confidence Group',common_norm=True,multiple='fill',alpha=1,palette='Paired',bw_adjust=20,ax=ax)

    #legend_elements = [Line2D([0], [0], color=color, label=label) for color, label in zip(kde.get_lines()[::len(confidence_df['Confidence Group'].unique())], confidence_df['Confidence Group'].unique())]
    #ax.legend(handles=legend_elements, loc='lower right')

    ax.set_xlabel('Attribution Score (Absolute Value)')
    #ax.set_xlabel('Rank')
    #ax.set_xlim(0,1)

"""
#plt.clf()
#fig,ax = plt.subplots(3,1,figsize=(7,6),sharey=True,sharex=True)
#fig.subplots_adjust(left=0.1,bottom=0.2,right=0.9,top=0.8,wspace=0.35,hspace=0.4)
def make_tf_hists(fig):
    #a = fig.subplots(3,1,sharey=True,sharex=True)
    #fig.subplots_adjust(left=0.1,bottom=0.2,right=0.9,top=0.8,wspace=0.35,hspace=0.4)
    for i,c in enumerate(confidence_groups.keys()):
        a_ = a[i]
        a_.set_xlim(-0.0002,0.001)
        sns.kdeplot(x=confidence_groups[c],fill=True,linewidth=1,ax=a_)
        a_.set_title(c)
        #a.hist(confidence_groups[c],bins=50)
        if i == 2:
            a_.set_xlabel('Attribution Score (Absolute Value)')

#fig.savefig('tf_hist.png')
"""

"""
print("AB, CD")
print(stats.kstest(confidence_groups['AB'],confidence_groups['CD'],alternative='less').pvalue)
print(stats.cramervonmises_2samp(confidence_groups['AB'],confidence_groups['CD']).pvalue)
print("AB, Other")
print(stats.kstest(confidence_groups['AB'],confidence_groups['Other'],alternative='less').pvalue)
print(stats.cramervonmises_2samp(confidence_groups['AB'],confidence_groups['Other']).pvalue)
print("CD, Other")
print(stats.kstest(confidence_groups['CD'],confidence_groups['Other'],alternative='less').pvalue)
print(stats.cramervonmises_2samp(confidence_groups['CD'],confidence_groups['Other']).pvalue)
"""

    


