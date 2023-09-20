import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from load_attribution_scores import get_attr_scores_one_run, ensembl_to_gene_name, get_attr_scores_avg_runs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)

attr_df = get_attr_scores_one_run()
attr_df = ensembl_to_gene_name(attr_df)

pkn_df = pd.read_pickle('pkn_moa_df.pkl').T

attr_df = ((attr_df-attr_df.mean())/attr_df.std())
"""
pca_embedding = PCA(n_components=30).fit_transform(attr_df)
pca_tsne_embedding = TSNE(n_components=3).fit_transform(pca_embedding)
"""

pkn_pca_embedding = PCA(n_components=30).fit_transform(pkn_df)
pkn_pca_tsne_embedding = TSNE(n_components=3).fit_transform(pkn_pca_embedding)

"""
plt.clf()
plt.scatter(pca_embedding[:,0],pca_embedding[:,1],marker='.',alpha=0.5)
plt.savefig("pca_attr_zscore.png")

plt.clf()
plt.scatter(pca_tsne_embedding[:,0],pca_tsne_embedding[:,1],marker='.',alpha=0.5)
plt.savefig("tsne_attr_zscore.png")
"""


plt.clf()
plt.scatter(pkn_pca_embedding[:,0],pkn_pca_embedding[:,1],marker='.',alpha=0.5)
plt.savefig("pkn_pca_attr_zscore.png")

plt.clf()
plt.scatter(pkn_pca_tsne_embedding[:,0],pkn_pca_tsne_embedding[:,1],marker='.',alpha=0.5)
plt.savefig("pkn_tsne_attr_zscore.png")



