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

tf_fc = base_path+"/save_model_tffc-fc_epochs100_batchsize128_enlr0.001_delr0.0001_moa1.0_rel_conn10_5-30_17.53.35/"
tf_shallow = base_path+"/save_model_tffc-shallow_epochs100_batchsize128_enlr0.001_delr0.01_moa1.0_rel_conn10_5-30_21.13.35/"
tf_gene = base_path+"/save_model_tffc-genefc_epochs100_batchsize128_enlr0.001_delr0.01_moa1.0_rel_conn10_5-30_21.13.36/"

shallow_fc = base_path+"/save_model_shallow-fc_epochs100_batchsize128_enlr0.001_delr0.0001_moa1.0_rel_conn10_5-31_9.18.27/"
shallow_shallow = base_path+"/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_moa1.0_rel_conn10_5-31_9.19.38/"
shallow_gene = base_path+"/save_model_shallow-genefc_epochs100_batchsize128_enlr0.01_delr0.01_moa1.0_rel_conn10_5-31_9.19.33/"

fc_fc= base_path+"/save_model_fc-fc_epochs100_batchsize128_enlr0.0001_delr0.0001_moa1.0_rel_conn10_5-30_17.51.19/"
fc_shallow = base_path+"/save_model_fc-shallow_epochs100_batchsize128_enlr0.0001_delr0.001_moa1.0_rel_conn10_5-30_17.52.55/"
fc_gene = base_path+"/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_rel_conn10_5-30_17.52.56/"

random_distances = make_random_ranks(fc_fc)

decoder = ['d_fc','d_shallow','d_gene']

fc_df = pd.DataFrame({'d_fc':calculate_consistency(fc_fc)[1],'d_gene':calculate_consistency(fc_gene)[1],'d_shallow':calculate_consistency(fc_shallow)[1]}).melt(value_vars=decoder,value_name='distance',var_name='decoder')
fc_df['encoder'] = 'e_fc'
print(fc_df)

shallow_df = pd.DataFrame({'d_fc':calculate_consistency(shallow_fc)[1],'d_gene':calculate_consistency(shallow_gene)[1],'d_shallow':calculate_consistency(shallow_shallow)[1]}).melt(value_vars=decoder,value_name='distance',var_name='decoder').dropna()
shallow_df['encoder'] = 'e_shallow'

tf_df = pd.DataFrame({'d_gene':calculate_consistency(tf_gene)[1], 'd_fc':calculate_consistency(tf_fc)[1], 'd_shallow':calculate_consistency(tf_shallow)[1]}).melt(value_vars=decoder,value_name='distance',var_name='decoder').dropna()
tf_df['encoder'] = 'e_tf'

enc_sorter = ['e_shallow','e_tf','e_fc']
dec_sorter = ['d_shallow','d_gene','d_fc']

df = pd.concat([shallow_df,tf_df,fc_df],axis=0)

df.decoder = df.decoder.astype("category")
df.decoder = df.decoder.cat.set_categories(dec_sorter)
df.encoder = df.encoder.astype("category")
df.encoder = df.encoder.cat.set_categories(enc_sorter)

print(df)













