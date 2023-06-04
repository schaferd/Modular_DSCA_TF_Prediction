import numpy as np
from scipy.stats import ttest_1samp, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

#TEST CORRELATION BETWEEN INPUT AND OUTPUT GENE EXPRESSION

base_path = "/nobackup/users/schaferd/ae_project_outputs/model_eval/"

tffc_fc = pd.read_pickle(base_path+"/save_model_tffc-fc_epochs100_batchsize128_enlr0.001_delr0.0001_moa1.0_rel_conn10_5-31_17.54.10/test_corrs.pkl")
tffc_shallow =pd.read_pickle(base_path+"/save_model_tffc-shallow_epochs100_batchsize128_enlr0.001_delr0.01_moa1.0_rel_conn10_6-1_10.41.11/test_corrs.pkl")
tffc_genefc =pd.read_pickle(base_path+"/save_model_tffc-genefc_epochs100_batchsize128_enlr0.001_delr0.01_moa1.0_rel_conn10_5-31_23.34.32/test_corrs.pkl")

shallow_fc = pd.read_pickle(base_path+"/save_model_shallow-fc_epochs100_batchsize128_enlr0.001_delr0.0001_moa1.0_rel_conn10_6-1_10.42.8/test_corrs.pkl")
shallow_shallow = pd.read_pickle(base_path+"/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_moa1.0_rel_conn10_6-1_10.43.21/test_corrs.pkl")
shallow_genefc = pd.read_pickle(base_path+"/save_model_shallow-genefc_epochs100_batchsize128_enlr0.01_delr0.01_moa1.0_rel_conn10_6-1_10.42.44/test_corrs.pkl")

fc_fc= pd.read_pickle(base_path+"/save_model_fc-fc_epochs100_batchsize128_enlr0.0001_delr0.0001_moa1.0_rel_conn10_5-31_17.51.41/test_corrs.pkl")
fc_shallow = pd.read_pickle(base_path+"/save_model_fc-shallow_epochs100_batchsize128_enlr0.0001_delr0.001_moa1.0_rel_conn10_5-31_17.52.35/test_corrs.pkl")
fc_genefc =pd.read_pickle(base_path+"/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_rel_conn10_5-31_17.52.2/test_corrs.pkl")


corr_dict = {'tf_fc':tffc_fc,'tf_shallow':tffc_shallow,'tf_gene':tffc_genefc, 'shallow_fc':shallow_fc, 'shallow_shallow':shallow_shallow, 'shallow_gene':shallow_genefc,'fc_fc':fc_fc,'fc_shallow':fc_shallow,'fc_gene':fc_genefc}

decoder = ['d_fc','d_shallow','d_gene']

fc_df = pd.concat([pd.DataFrame({'d_fc':fc_fc}),pd.DataFrame({'d_shallow':fc_shallow}),pd.DataFrame({'d_gene':fc_genefc})],axis=1)

fc_df = fc_df.melt(value_vars=decoder,value_name='Corr',var_name='decoder').dropna()
fc_df['encoder'] = 'e_fc'

shallow_df = pd.concat([pd.DataFrame({'d_fc':shallow_fc}),pd.DataFrame({'d_shallow':shallow_shallow}),pd.DataFrame({'d_gene':shallow_genefc})],axis=1)
shallow_df = shallow_df.melt(value_vars=decoder,value_name='Corr',var_name='decoder').dropna()
shallow_df['encoder'] = 'e_shallow'

tffc_df = pd.concat([pd.DataFrame({'d_fc':tffc_fc}),pd.DataFrame({'d_shallow':tffc_shallow}),pd.DataFrame({'d_gene':tffc_genefc})],axis=1)
tffc_df = tffc_df.melt(value_vars=decoder,value_name='Corr',var_name='decoder').dropna()
tffc_df['encoder'] = 'e_tf'




enc_sorter = ['e_shallow','e_tf','e_fc']
dec_sorter = ['d_shallow','d_gene','d_fc']


df = pd.concat([fc_df,shallow_df,tffc_df],axis=0)

df.decoder = df.decoder.astype("category")
df.decoder = df.decoder.cat.set_categories(dec_sorter)
df.encoder = df.encoder.astype("category")
df.encoder = df.encoder.cat.set_categories(enc_sorter)


###PVALUES FOR PAPER
shallow_decoder = tffc_shallow+shallow_shallow+fc_shallow
gene_decoder = tffc_genefc+shallow_genefc+fc_genefc
stat, pval = ttest_ind(shallow_decoder,gene_decoder)
print('shallow vs. gene decoder',pval)

fc_decoder = tffc_fc+shallow_fc+fc_fc
sparse_decoder=gene_decoder+shallow_decoder
stat, pval = ttest_ind(fc_decoder,sparse_decoder)
print('fc vs. sparse decoder',pval)


print(df)
model = ols('Corr ~ C(encoder) + C(decoder) + C(encoder):C(decoder)',data=df).fit()
print(sm.stats.anova_lm(model,typ=2))
d_results = sm.stats.multicomp.pairwise_tukeyhsd(endog=df['Corr'],groups=df['decoder'],alpha=0.05)
e_results = sm.stats.multicomp.pairwise_tukeyhsd(endog=df['Corr'],groups=df['encoder'],alpha=0.05)
print(e_results)
print(e_results.pvalues)

print(d_results)
print(d_results.pvalues)
