import numpy as np
from scipy.stats import ttest_1samp, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#KO ROC AUC DATA

base_path = "/nobackup/users/schaferd/ae_project_outputs/model_eval/l2norm_test_diff/"

tffc_fc = pd.read_pickle(base_path+"__tffc-fc_epochs100_batchsize128_enlr0.001_delr0.0001_del20.0001_enl21e-05_moa1.0_rel_conn10_2-8_12.58.19/aucs.pkl") 
tffc_shallow = pd.read_pickle(base_path+"__tffc-shallow_epochs100_batchsize128_enlr0.001_delr0.01_del21e-05_enl21e-05_moa1.0_rel_conn10_2-8_20.44.1/aucs.pkl")
tffc_genefc = pd.read_pickle(base_path+"/__tffc-genefc_epochs100_batchsize128_enlr0.001_delr0.01_del21e-05_enl21e-05_moa1.0_rel_conn10_2-8_18.38.21/aucs.pkl")

shallow_fc = pd.read_pickle(base_path+"__shallow-fc_epochs100_batchsize128_enlr0.001_delr0.0001_del20.0001_enl20.0001_moa1.0_rel_conn10_2-9_8.58.20/aucs.pkl")
shallow_shallow = pd.read_pickle(base_path+"__shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.0001_enl20.0001_moa1.0_rel_conn10_2-9_8.58.28/aucs.pkl")
shallow_genefc = pd.read_pickle(base_path+"__shallow-genefc_epochs100_batchsize128_enlr0.01_delr0.01_del20.0001_enl20.0001_moa1.0_rel_conn10_2-9_8.58.20/aucs.pkl")

fc_fc= pd.read_pickle(base_path+"__fc-fc_epochs100_batchsize128_enlr0.0001_delr0.0001_del20.0001_enl20.0001_moa1.0_rel_conn10_2-8_12.56.13/aucs.pkl")
fc_shallow = pd.read_pickle(base_path+"__fc-shallow_epochs100_batchsize128_enlr0.0001_delr0.001_del21e-05_enl20.0001_moa1.0_rel_conn10_2-8_12.57.4/aucs.pkl")
fc_genefc = pd.read_pickle(base_path+"__fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0001_moa1.0_rel_conn10_2-8_12.56.37/aucs.pkl")

auc_dict = {'tf_fc':tffc_fc,'tf_shallow':tffc_shallow,'tf_gene':tffc_genefc, 'shallow_fc':shallow_fc, 'shallow_shallow':shallow_shallow, 'shallow_gene':shallow_genefc,'fc_fc':fc_fc,'fc_shallow':fc_shallow,'fc_gene':fc_genefc}

decoder = ['d_fc','d_shallow','d_gene']

fc_df = pd.concat([pd.DataFrame({'d_fc':fc_fc}),pd.DataFrame({'d_shallow':fc_shallow}),pd.DataFrame({'d_gene':fc_genefc})],axis=1)

fc_df = fc_df.melt(value_vars=decoder,value_name='AUC',var_name='decoder').dropna()
fc_df['encoder'] = 'e_fc'

shallow_df = pd.concat([pd.DataFrame({'d_fc':shallow_fc}),pd.DataFrame({'d_shallow':shallow_shallow}),pd.DataFrame({'d_gene':shallow_genefc})],axis=1)
shallow_df = shallow_df.melt(value_vars=decoder,value_name='AUC',var_name='decoder').dropna()
shallow_df['encoder'] = 'e_shallow'

tffc_df = pd.concat([pd.DataFrame({'d_fc':tffc_fc}),pd.DataFrame({'d_shallow':tffc_shallow}),pd.DataFrame({'d_gene':tffc_genefc})],axis=1)
tffc_df = tffc_df.melt(value_vars=decoder,value_name='AUC',var_name='decoder').dropna()
tffc_df['encoder'] = 'e_tf'


enc_sorter = ['e_shallow','e_tf','e_fc']
dec_sorter = ['d_shallow','d_gene','d_fc']

df = pd.concat([shallow_df,tffc_df,fc_df],axis=0)

df.decoder = df.decoder.astype("category")
df.decoder = df.decoder.cat.set_categories(dec_sorter)
df.encoder = df.encoder.astype("category")
df.encoder = df.encoder.cat.set_categories(enc_sorter)

print(df.decoder)
print(df.encoder)

