import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy import stats
from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)
from load_attribution_scores import get_attr_scores_one_run, ensembl_to_gene_name, get_attr_scores_avg_runs
from sklearn.cluster import AgglomerativeClustering

n_clusters = 6

filtered_pkn = pd.read_csv('filtered_pkn.tsv',sep='\t',index_col=0)

attr_df = get_attr_scores_one_run()
attr_df = ensembl_to_gene_name(attr_df)

attr_df = ((attr_df)/attr_df.std())

clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(attr_df)
print(len(clustering.labels_))

tf_to_index_dict = {}
index_to_tf_dict = {}
#assign each tf a number 
for i,tf in enumerate(attr_df.columns):
    tf_to_index_dict[tf] = i
    index_to_tf_dict[i] = tf

cluster_tfs_dict = {cluster:[] for cluster in range(n_clusters)}
cluster_tf_freq = pd.DataFrame(np.zeros((len(attr_df.columns),n_clusters)),index=attr_df.columns)

for i in range(n_clusters):
    #create a hist of tfs that are in cluster i
    genes = attr_df.index[clustering.labels_ == i]
    print(i,genes)
    for g in genes:
        tfs = filtered_pkn[filtered_pkn['target'] == g]['tf']
        for tf in tfs:
            cluster_tf_freq.at[tf,i] += 1

cluster_tf_freq = cluster_tf_freq/cluster_tf_freq.sum(axis=0)
        

sns.heatmap(cluster_tf_freq,cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True))
plt.savefig('cluster_tf_freq_heatmap.png')

"""
plt.clf()
fig,ax = plt.subplots(n_clusters,1,figsize=(5,15),sharey=True,sharex=True)
fig.subplots_adjust(left=0.1,bottom=0.2,right=0.9,top=0.8,wspace=0.35,hspace=0.8)
for i in range(n_clusters):
    a = ax[i]
    #sns.distplot(cluster_tfs_dict[i],kde=True,hist=True,ax=a,bins=500)
    a.set_title('Cluster '+str(i))
    a.hist(cluster_tfs_dict[i],bins=108)
    if i == n_clusters-1:
        a.set_xlabel('TF (index)')

fig.savefig('cluster_hist.png')
"""
