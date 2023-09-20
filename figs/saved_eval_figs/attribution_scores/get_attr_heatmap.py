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
from load_attribution_scores import get_attr_scores_one_run, ensembl_to_gene_name, get_attr_scores_avg_runs
from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)

attr_df = get_attr_scores_one_run()

gene_to_ensembl = {}
ensembl_to_gene = {}

for g in attr_df.index:
        try:
                gene_id = ensembl_data.gene_name_of_gene_id(g)
                gene_to_ensembl[gene_id] = g 
                ensembl_to_gene[g] = gene_id
        except:
                ensembl_to_gene[g] = None

attr_df = attr_df.rename(index=ensembl_to_gene)
attr_df = attr_df
#attr_df = ((attr_df-attr_df.mean())/attr_df.std())
attr_df = ((attr_df)/attr_df.std()).T
#attr_df = attr_df.fillna(0)

#attr_df.to_pickle('attr_df_tf_zscore.pkl')
#raise ValueError()

n = 10
top_genes = set()

for tf in attr_df.index:
    row_ranks = attr_df.loc[tf,:].abs().argsort()
    top_n = row_ranks[row_ranks > len(row_ranks)-n]
    top_genes = top_genes.union(set(top_n.index))

top_genes = list(top_genes)
top_genes.sort()


attr_df_filtered = attr_df.loc[:,top_genes]
#attr_df_filtered = attr_df
"""
attr_df_filtered[attr_df_filtered.abs() < 3] = 0
attr_df_filtered[attr_df_filtered >= 3] = 1
attr_df_filtered[attr_df_filtered <= -3] = -1
"""

#CLUSTER MAP
plt.clf()
fig,ax = plt.subplots()
sns.diverging_palette(220, 20, as_cmap=True)
fig.set_figwidth(18)
fig.set_figheight(15)
ax = sns.clustermap(attr_df_filtered,center=0,figsize=(14,12),cmap='vlag',vmin=-5,vmax=5)
#ax = sns.clustermap(attr_df_filtered,center=0,figsize=(14,12),cmap='vlag',vmin=-1,vmax=1)
plt.savefig('tf_zscored_attr_heatmap.png')
"""
#HEATMAP WITH PKN ORDERING
row_ordering = pd.read_pickle('pkn_row_indices.pkl')
col_ordering = pd.read_pickle('pkn_col_indices.pkl')
pkn_cluster_attr_df = attr_df_filtered.loc[row_ordering,col_ordering]
fig,ax = plt.subplots()
fig.set_figwidth(18)
fig.set_figheight(15)
sns.heatmap(pkn_cluster_attr_df,center=0,ax=ax,cmap='vlag',vmin=-5,vmax=5)
fig.savefig('pkn_clustered_gene_zscored_restricted_attr_heatmap.png')
"""


