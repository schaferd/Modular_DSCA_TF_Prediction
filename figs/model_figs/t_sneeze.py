import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
from scipy.stats import ttest_1samp, ttest_ind, kendalltau
from scipy import stats
from sklearn import metrics
from sklearn.manifold import TSNE
import sys
import os

consistency_eval_path = '/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/'
sys.path.insert(1,consistency_eval_path)
from check_consistency_ko import calculate_consistency, make_random_ranks

base_path = "/nobackup/users/schaferd/ae_project_outputs/model_eval//"

tf_fc = base_path+"/__tffc-fc_epochs100_batchsize128_enlr0.001_delr0.0001_moa1.0_rel_conn10_2-10_16.14.43/"
tf_shallow = base_path+"/__tffc-shallow_epochs100_batchsize128_enlr0.001_delr0.01_moa1.0_rel_conn10_2-11_15.50.53/"
tf_gene = base_path+"/__tffc-genefc_epochs100_batchsize128_enlr0.001_delr0.01_moa1.0_rel_conn10_2-10_21.56.47/"

shallow_fc =base_path+"/__shallow-fc_epochs100_batchsize128_enlr0.001_delr0.0001_moa1.0_rel_conn10_2-11_15.50.58/"
shallow_shallow = base_path+"/__shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_moa1.0_rel_conn10_2-11_15.51.40/"
shallow_gene = base_path+"/__shallow-genefc_epochs100_batchsize128_enlr0.01_delr0.01_moa1.0_rel_conn10_2-11_15.51.25/"

fc_fc= base_path+"/__fc-fc_epochs100_batchsize128_enlr0.0001_delr0.0001_moa1.0_rel_conn10_2-10_16.11.55/"
fc_shallow =base_path+"/__fc-shallow_epochs100_batchsize128_enlr0.0001_delr0.001_moa1.0_rel_conn10_2-10_16.14.5/"
fc_gene = base_path+"/__fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_rel_conn10_2-10_16.13.22/"

cols = ['fc_fc', 'fc_g','fc_s','s_fc','s_g','s_s','t_g','t_fc','t_s']

num_trials = calculate_consistency(fc_fc)[0].shape[0]
print( calculate_consistency(fc_fc)[0].shape)

sample_ids = []
for col in cols:
    for i in range(num_trials):
        sample_ids.append(col)
print(sample_ids)


data = np.array([*calculate_consistency(fc_fc)[0],*calculate_consistency(fc_gene)[0],*calculate_consistency(fc_shallow)[0],*calculate_consistency(shallow_fc)[0],*calculate_consistency(shallow_gene)[0],*calculate_consistency(shallow_shallow)[0],*calculate_consistency(tf_gene)[0],*calculate_consistency(tf_fc)[0],*calculate_consistency(tf_shallow)[0]])

distance_matrix = None
kendall_matrix = None

def get_dists(data,sample):
    dists = []
    for row in data:
        dists.append(np.sqrt(np.sum(np.square(row-sample))))
    return dists

def get_kends(data,sample):
    kends = []
    for row in data:
        kends.append(1-(2/(1+kendalltau(row,sample)[0]+0.0001)))
    return kends 

for sample in data:
    dist = get_dists(data, sample)
    kend = get_kends(data, sample)
    if distance_matrix is None:
        distance_matrix = dist
        kendall_matrix = kend
    else:
        distance_matrix = np.vstack((distance_matrix,dist))
        kendall_matrix = np.vstack((kendall_matrix,kend))

print(distance_matrix)
print(kendall_matrix)


#print(data)
#print(data.shape)


kend_embedding = pd.DataFrame(TSNE(n_components=2,learning_rate='auto',init='random',perplexity=5).fit_transform(kendall_matrix),columns=['tsne1','tsne2'])
dist_embedding = pd.DataFrame(TSNE(n_components=2,learning_rate='auto',init='random',perplexity=5).fit_transform(distance_matrix),columns=['tsne1','tsne2'])
kend_embedding['id']=sample_ids
dist_embedding['id']=sample_ids

#print(embedding)
#print(kend_embedding)

sns.scatterplot(
        x='tsne1',y='tsne2',
        hue='id',
        data=dist_embedding,
        legend='full',
        )
plt.title('Consistency T-SNE')
plt.savefig('dist_tsneeze.png')


plt.clf()
sns.scatterplot(x='tsne1',y='tsne2',hue='id',        data=kend_embedding,        legend='full',)

plt.savefig('kend_tsneeze.png')
