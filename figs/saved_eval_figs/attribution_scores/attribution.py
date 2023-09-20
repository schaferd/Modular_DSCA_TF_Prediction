import pandas as pd
from functools import reduce
from scipy import stats
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
from statsmodels.stats.multitest import multipletests
from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)

"""
base_path = '/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/'
fc_g_path = [base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.56/',base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/']
input_genes = pd.read_pickle(base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/fold0_cycle0/ko_activities_cycle0_fold0/input_genes.pkl')

pknAB = pd.read_csv('pknAB.csv',sep='\t')
pknCD = pd.read_csv('pknCD.csv',sep='\t')
print(pknAB)

fc_g_attr_files = []

for path in fc_g_path:
    fold_paths = []
    fold_paths = fold_paths+[path+'/'+p for p in os.listdir(path) if 'fold' in p]
    act_path = []
    for p in fold_paths:
        act_path = act_path + [p+'/'+i for i in os.listdir(p) if 'ko_activities' in i]
        for p2 in act_path:
            fc_g_attr_files = fc_g_attr_files+[p2+'/'+f for f in os.listdir(p2) if 'tf_attr_dict.pkl' in f]

attr_dicts = [pd.read_pickle(f) for f in fc_g_attr_files]

tf_attr_dict = {}
for model_attr in attr_dicts:
    for tf in model_attr.keys():
        if tf not in tf_attr_dict:
            tf_attr_dict[tf] = [np.array([model_attr[tf]])]
        else:
            tf_attr_dict[tf].append(np.array([model_attr[tf]]))

tf_attr_dict = {tf:np.vstack(tf_attr_dict[tf]) for tf in tf_attr_dict.keys()}
tf_attr_mean_dict = {tf:np.mean(np.mean(tf_attr_dict[tf],axis=0),axis=0) for tf in tf_attr_dict.keys()}

attr_df = pd.DataFrame(tf_attr_mean_dict,index=input_genes)
print(attr_df)

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
rand_df = pd.DataFrame(np.random.rand(*attr_df.shape),columns=attr_df.columns,index=attr_df.index)


CD_genes = list(set(list(pknCD['target'])).intersection(set(list(attr_df.columns))))
AB_genes = list(set(list(pknAB['target'])).intersection(set(list(attr_df.columns))))
attr_df_CD_filtered = attr_df[CD_genes]
attr_df_AB_filtered = attr_df[AB_genes]
randAB_df = pd.DataFrame(np.random.rand(*attr_df_AB_filtered.shape),columns=attr_df_AB_filtered.columns,index=attr_df_AB_filtered.index)
randCD_df = pd.DataFrame(np.random.rand(*attr_df_CD_filtered.shape),columns=attr_df_CD_filtered.columns,index=attr_df_CD_filtered.index)


N = np.concatenate([np.array([50]),np.arange(100,len(attr_df.columns),100),np.array([len(attr_df.columns)-1])])
#N = [100,400,3700]
no_samples = 100
N_sig_AB = []
N_sig_CD = []


for n in N:
    print(n)
    tf_sig_dictAB = {}
    tf_sig_dictCD = {}

    #min_sig_diff = 100
    #min_sig_diff_num = 0
    #min_sig_diff_rand_dist = []

    for tf,row in attr_df.iterrows():
        #ranksAB = attr_df_AB_filtered.loc[tf,:].abs().rank()
        #ranksCD = attr_df_CD_filtered.loc[tf,:].abs().rank()
        ranks = attr_df.loc[tf,:].abs().rank()

        #topAB = attr_df_AB_filtered.loc[tf,:][ranksAB > max(ranksAB)-n]
        #topCD = attr_df_CD_filtered.loc[tf,:][ranksCD > max(ranksCD)-n]
        topAB = attr_df.loc[tf,:][ranks > max(ranks)-n]
        topCD = topAB
        
        #rand_topAB = [randAB_df.loc[tf,:].sample(n) for i in range(no_samples)]
        #rand_topCD = [randCD_df.loc[tf,:].sample(n) for i in range(no_samples)]
        rand_topAB = [rand_df.loc[tf,:].sample(n) for i in range(no_samples)]
        rand_topCD = rand_topAB

        relationshipsCD = set(pknCD[pknCD['source'] == tf]['target']).intersection(set(CD_genes))
        relationshipsAB = set(pknAB[pknAB['source'] == tf]['target']).intersection(set(AB_genes))

        if len(relationshipsAB) > 0:
            rel_in_top = relationshipsAB.intersection(set(topAB.index))
            rand_rel_in_top = np.array([len(relationshipsAB.intersection(set(df.index))) for df in rand_topAB])
            #stat, pval = stats.ttest_1samp(rand_rel_in_top,popmean=len(rel_in_top),alternative='less')
            pval = (rand_rel_in_top >= len(rel_in_top)).sum()/len(rand_rel_in_top)
            tf_sig_dictAB[tf] = [pval]
        else:
            tf_sig_dictAB[tf] = [1]
        if len(relationshipsCD) > 0:
            rel_in_top = relationshipsCD.intersection(set(topCD.index))
            rand_rel_in_top = np.array([len(relationshipsCD.intersection(set(df.index))) for df in rand_topCD])
            #stat, pval = stats.ttest_1samp(rand_rel_in_top,popmean=len(rel_in_top),alternative='less')
            pval = (rand_rel_in_top >= len(rel_in_top)).sum()/len(rand_rel_in_top)
            tf_sig_dictCD[tf] = [pval]
            #if (pval < 0.05 and len(rel_in_top) - np.mean(rand_rel_in_top) < min_sig_diff):
            #    min_sig_diff = len(rel_in_top) - np.mean(rand_rel_in_top)
            #    min_sig_diff_num = len(rel_in_top)
            #    min_sig_diff_rand_dist = rand_rel_in_top
        else:
            tf_sig_dictCD[tf] = [1]
    #plt.clf()
    #sns.swarmplot(min_sig_diff_rand_dist)
    #plt.axhline(min_sig_diff_num)
    #plt.axhline(np.mean(min_sig_diff_rand_dist))
    #plt.savefig("min_diff_hist_"+str(n)+".png")
    #plt.clf()

    tfs = []
    pvalsAB = []
    pvalsCD = []
    for tf in tf_sig_dictAB.keys():
        tfs.append(tf)
        pvalsAB.append(tf_sig_dictAB[tf][0])
        pvalsCD.append(tf_sig_dictCD[tf][0])
    pvalsAB = multipletests(pvalsAB,alpha=0.05,method='fdr_bh')[1]
    pvalsCD = multipletests(pvalsCD,alpha=0.05,method='fdr_bh')[1]
    tf_sig_dictAB = {tf:pvalsAB[i] for i,tf in enumerate(tfs)}
    tf_sig_dictCD = {tf:pvalsCD[i] for i,tf in enumerate(tfs)}

    N_sig_AB.append(pd.DataFrame(tf_sig_dictAB,index=[n]))
    N_sig_CD.append(pd.DataFrame(tf_sig_dictCD,index=[n]))

N_sig_dfAB = pd.concat(N_sig_AB)
N_sig_dfCD = pd.concat(N_sig_CD)

print(N_sig_dfAB)
print(N_sig_dfCD)

per_sig_dictAB = {}
per_sig_dictCD = {}
alpha = 0.05

for n in N_sig_dfAB.index:
    rowAB = N_sig_dfAB.loc[n,:]
    rowCD = N_sig_dfCD.loc[n,:]

    sigAB = len(rowAB[rowAB < alpha])/len(rowAB.index)
    sigCD = len(rowCD[rowCD < alpha])/len(rowCD.index)

    per_sig_dictAB[n] = [sigAB]
    per_sig_dictCD[n] = [sigCD]

per_sig_dfAB = pd.DataFrame(per_sig_dictAB)
per_sig_dfCD = pd.DataFrame(per_sig_dictCD)

per_sig_dfAB['pkn'] = 'AB'
per_sig_dfCD['pkn'] = 'CD'

per_sig_df = pd.concat([per_sig_dfAB,per_sig_dfCD],axis=0)
print(per_sig_df)
per_sig_df = per_sig_df.melt(value_vars=N,id_vars=['pkn'],value_name='percent_sig',var_name='top_n')
print(per_sig_df)

#per_sig_df = pd.concat([per_sig_df,pd.read_pickle('per_sig_df.pkl')],axis=0)
#print(per_sig_df)
"""

per_sig_df = pd.read_pickle('per_sig_df_sample100.pkl')
print(per_sig_df)
per_sig_df = per_sig_df.rename(columns={'pkn':'PKN'})
per_sig_df['percent_sig'] = per_sig_df['percent_sig']*100

markers = ['X','o']
colors = ['black','darkgrey']

fig,ax= plt.subplots()

sns.scatterplot(x='top_n',y='percent_sig',data=per_sig_df[per_sig_df['PKN'] =='AB'],markers=True,color='black',marker='s',ax=ax,legend=True,label='AB')
sns.lineplot(x='top_n',y='percent_sig',data=per_sig_df[per_sig_df['PKN'] =='AB'],color='black',ax=ax,legend=False)

sns.scatterplot(x='top_n',y='percent_sig',data=per_sig_df[per_sig_df['PKN'] =='CD'],ax=ax,color='slategrey',marker='o',legend=True,label='CD')
sns.lineplot(x='top_n',y='percent_sig',data=per_sig_df[per_sig_df['PKN'] =='CD'],color='slategrey',ax=ax,legend=False)

ax.set_title('Percent of TFs with Sig. No. of Genes in\nBoth PKN and Top n Ranked Gene Attribution Scores')
ax.set_ylabel('Percent of TFs')
ax.set_xlabel('Top n Ranked Genes')
ax.set_ylim(-5,100)
#fig.savefig('percent_sig_sample'+str(no_samples)+'.png')
fig.savefig('percent_sig_sample.png')



#per_sig_df.to_pickle('per_sig_df_sample'+str(no_samples)+'.pkl')

    
    




#fig,ax = plt.subplots(figsize=(20,40))
#sns.heatmap(df,cmap='vlag',center=0,ax=ax,vmin=-0.01,vmax=0.01)
#fig.savefig('attr_heatmap.png')
