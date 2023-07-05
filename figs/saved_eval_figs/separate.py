import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

s_s_train = np.array(pd.read_pickle('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/shallow_shallow/agg_data.pkl.activities.pkl'))
np.random.shuffle(s_s_train)
s_s_train = s_s_train[:200]
fc_g_train = np.array(pd.read_pickle('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/fc_g/agg_data.pkl.activities.pkl'))
np.random.shuffle(fc_g_train)
fc_g_train = fc_g_train[:200]
s_s_encode = np.array(pd.read_pickle('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/shallow_shallow/combined_processed_df.pkl.activities.pkl'))
fc_g_encode = np.array(pd.read_pickle('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/fc_g/combined_processed_df.pkl.activities.pkl'))

s_s_encode = s_s_encode.reshape(-1,s_s_encode.shape[-1])
fc_g_encode = fc_g_encode.reshape(-1,fc_g_encode.shape[-1])
fc_g_train = fc_g_train.reshape(-1,fc_g_train.shape[-1])
s_s_train = s_s_train.reshape(-1,s_s_train.shape[-1])

no_train = s_s_train.shape[0]+fc_g_train.shape[0]
no_encode = s_s_encode.shape[0]+fc_g_encode.shape[0]

results = np.vstack([s_s_encode,fc_g_encode,s_s_train,fc_g_train])

#train_labels = np.repeat('train', no_train)
#encode_labels = np.repeat('encode',no_encode)
#labels = np.hstack([encode_labels,train_labels])

#train_colors = np.repeat('orange',s_s_train.shape[0]+fc_g_train.shape[0])
#encode_colors = np.repeat('purple',s_s_encode.shape[0]+fc_g_encode.shape[0])
#colors = np.hstack([encode_colors,train_colors])


pca_embedding = PCA(n_components=20).fit_transform(results)

pca_tsne_embedding = TSNE(n_components=2).fit_transform(pca_embedding)

print(pca_embedding)
plt.scatter(pca_embedding[no_encode:,0],pca_embedding[no_encode:,1],marker='.')
plt.scatter(pca_embedding[:no_encode,0],pca_embedding[:no_encode,1],marker='.')
plt.savefig('pca.png')
plt.clf()

plt.scatter(pca_tsne_embedding[no_encode:,0],pca_tsne_embedding[no_encode:,1],marker='.')
plt.scatter(pca_tsne_embedding[:no_encode,0],pca_tsne_embedding[:no_encode,1],marker='.')
plt.savefig('tsne.png')
