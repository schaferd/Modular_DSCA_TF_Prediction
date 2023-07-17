import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

input_data = pd.read_csv('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/treated_relevant_samples.csv',sep='\t',index_col=0)
ko_tf_df = pd.read_pickle('ko_tf_df.pkl')

input_data = input_data.loc[ko_tf_df.index,:]

pca_embedding = PCA(n_components=30).fit_transform(input_data.to_numpy())
pca_tsne_embedding = TSNE(n_components=2).fit_transform(pca_embedding)

pca_df = pd.DataFrame(pca_embedding[:,:2],index=input_data.index) 
tsne_df = pd.DataFrame(pca_tsne_embedding, index=input_data.index)


ko_tf_df = ko_tf_df.loc[input_data.index,:]
pca_df = pd.concat([pca_df,ko_tf_df],axis=1)
print(pca_df)

tsne_df = pd.concat([tsne_df,ko_tf_df],axis=1)
print(tsne_df)


fig,ax = plt.subplots(2,2)
fig.set_figwidth(10)
fig.set_figheight(10)

sns.scatterplot(x=0,y=1,hue='ae',data=tsne_df,ax=ax[0][0])
ax[0][0].set_title('AE TSNE')

sns.scatterplot(x=0,y=1,hue='ae',data=pca_df,ax=ax[0][1])
ax[0][1].set_title('AE PCA')


sns.scatterplot(x=0,y=1,hue='viper',data=tsne_df,ax=ax[1][0])
ax[1][0].set_title('VIPER TSNE')

sns.scatterplot(x=0,y=1,hue='viper',data=pca_df,ax=ax[1][1])
ax[1][1].set_title('VIPER PCA')

fig.savefig('pca_tsne_input.png')
