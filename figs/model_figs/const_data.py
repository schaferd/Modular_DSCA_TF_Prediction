import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns
import pandas as pd
import numpy as np

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

random_distances = make_random_ranks(fc_fc)

decoder = ['d_fc','d_shallow','d_gene']

fc_df = pd.DataFrame({'d_fc':calculate_consistency(fc_fc),'d_gene':calculate_consistency(fc_gene),'d_shallow':calculate_consistency(fc_shallow)}).melt(value_vars=decoder,value_name='distance',var_name='decoder')
fc_df['encoder'] = 'e_fc'
print(fc_df)

shallow_df = pd.DataFrame({'d_fc':calculate_consistency(shallow_fc),'d_gene':calculate_consistency(shallow_gene),'d_shallow':calculate_consistency(shallow_shallow)}).melt(value_vars=decoder,value_name='distance',var_name='decoder').dropna()
shallow_df['encoder'] = 'e_shallow'

tf_df = pd.DataFrame({'d_gene':calculate_consistency(tf_gene), 'd_fc':calculate_consistency(tf_fc), 'd_shallow':calculate_consistency(tf_shallow)}).melt(value_vars=decoder,value_name='distance',var_name='decoder').dropna()
tf_df['encoder'] = 'e_tf'

enc_sorter = ['e_shallow','e_tf','e_fc']
dec_sorter = ['d_shallow','d_gene','d_fc']

df = pd.concat([shallow_df,tf_df,fc_df],axis=0)

df.decoder = df.decoder.astype("category")
df.decoder = df.decoder.cat.set_categories(dec_sorter)
df.encoder = df.encoder.astype("category")
df.encoder = df.encoder.cat.set_categories(enc_sorter)

print(df)













