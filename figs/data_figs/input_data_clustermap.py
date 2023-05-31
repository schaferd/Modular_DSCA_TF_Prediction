import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data_file = '/home/schaferd/ae_project/input_data_processing/hdf_agg_data.pkl'
input_data = pd.read_pickle(data_file).sample(200,axis=0).sample(200,axis=1)
print(input_data)

g1 = sns.clustermap(input_data,cmap='RdBu_r', vmin=-2, vmax=2)
g1.cax.set_visible(False)
#g1.fig.subplots_adjust(right=0.8,top=0.8)
#g1.ax_cbar.set_position((0.9,0.2,0.03,0.4))
g1.fig.savefig('input_data200_clustermap.png',dpi=300)
