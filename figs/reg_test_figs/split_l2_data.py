import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import tukey_hsd

base_path = '/nobackup/users/schaferd/ae_project_outputs/reg_tests/'

fc_3_g_3 = base_path+'l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.001_enl20.001_moa1.0_rel_conn10_2-12_11.31.45/'
fc_3_g_5_4 = base_path+'split_l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0005_enl20.001_moa1.0_rel_conn10_2-15_10.19.16/'
fc_3_g_4 = base_path+'split_l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.001_moa1.0_rel_conn10_2-14_8.3.9/'
fc_3_g_5_5 = base_path+'split_l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del25e-05_enl20.001_moa1.0_rel_conn10_2-15_13.9.29/'
fc_3_g_5 = base_path+'split_l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del21e-05_enl20.001_moa1.0_rel_conn10_2-14_8.3.9/'

fc_5_4_g_3 = base_path+'split_l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.001_enl20.0005_moa1.0_rel_conn10_2-15_13.15.59/'
fc_5_4_g_5_4 = base_path+'split_l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0005_enl20.0005_moa1.0_rel_conn10_2-15_7.58.42/'
fc_5_4_g_4 = base_path+'split_l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_2-15_9.50.3/'
fc_5_4_g_5_5 = base_path+'split_l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del25e-05_enl20.0005_moa1.0_rel_conn10_2-15_9.50.9/'
fc_5_4_g_5 = base_path+'split_l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del21e-05_enl20.0005_moa1.0_rel_conn10_2-15_21.39.33/'

fc_4_g_3 = base_path+'split_l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.001_enl20.0001_moa1.0_rel_conn10_2-13_20.23.39/'
fc_4_g_5_4 = base_path+'split_l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0005_enl20.0001_moa1.0_rel_conn10_2-15_13.53.24/'
fc_4_g_4 = base_path+'l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0001_moa1.0_rel_conn10_2-12_11.31.35/'
fc_4_g_5_5 = base_path+'split_l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del25e-05_enl20.0001_moa1.0_rel_conn10_2-15_13.16.58/'
fc_4_g_5 = base_path+'split_l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del21e-05_enl20.0001_moa1.0_rel_conn10_2-13_21.0.40/'

fc_5_g_3 = base_path+'split_l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.001_enl21e-05_moa1.0_rel_conn10_2-13_20.23.31/'
fc_5_g_5_4 = base_path+'l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0005_enl21e-05_moa1.0_rel_conn10_3-5_12.11.17/'
fc_5_g_4 = base_path+'split_l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl21e-05_moa1.0_rel_conn10_2-13_20.23.31/'
fc_5_g_5_5 = base_path+'l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del25e-05_enl21e-05_moa1.0_rel_conn10_3-5_12.11.17/'
fc_5_g_5 = base_path+'l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del21e-05_enl21e-05_moa1.0_rel_conn10_2-12_11.31.31/'

labels = ['g5','g55','g4','g54','g3']

filename = 'aucs.pkl'
fc_3_auc = [pd.read_pickle(fc_3_g_3+filename),pd.read_pickle(fc_3_g_5_4+filename),pd.read_pickle(fc_3_g_4+filename),pd.read_pickle(fc_3_g_5_5+filename),pd.read_pickle(fc_3_g_5+filename)]
fc_3_auc.reverse()
fc_3_auc_df = pd.DataFrame({labels[i]:fc_3_auc[i] for i in range(len(labels))}).melt(value_vars=labels,value_name='AUC',var_name='decoder')
fc_3_auc_df['encoder'] = 'fc3'

#fc_4_auc = [pd.read_pickle(fc_4_g_3+filename),pd.read_pickle(fc_4_g_4+filename),pd.read_pickle(fc_4_g_5+filename)]
fc_5_4_auc = [pd.read_pickle(fc_5_4_g_3+filename),pd.read_pickle(fc_5_4_g_5_4+filename),pd.read_pickle(fc_5_4_g_4+filename),pd.read_pickle(fc_5_4_g_5_5+filename),pd.read_pickle(fc_5_4_g_5+filename)]
fc_5_4_auc.reverse()
fc_5_4_auc_df = pd.DataFrame({labels[i]:fc_5_4_auc[i] for i in range(len(labels))}).melt(value_vars=labels,value_name='AUC',var_name='decoder')
fc_5_4_auc_df['encoder'] = 'fc54'

fc_4_auc= [pd.read_pickle(fc_4_g_3+filename),pd.read_pickle(fc_4_g_5_4+filename),pd.read_pickle(fc_4_g_4+filename),pd.read_pickle(fc_3_g_5_5+filename),pd.read_pickle(fc_4_g_5+filename)]
fc_4_auc.reverse()
fc_4_auc_df = pd.DataFrame({labels[i]:fc_4_auc[i] for i in range(len(labels))}).melt(value_vars=labels,value_name='AUC',var_name='decoder')
fc_4_auc_df['encoder'] = 'fc4'

fc_5_auc = [pd.read_pickle(fc_5_g_3+filename),pd.read_pickle(fc_5_g_5_4+filename),pd.read_pickle(fc_5_g_4+filename),pd.read_pickle(fc_5_g_5_5+filename),pd.read_pickle(fc_5_g_5+filename)]
fc_5_auc.reverse()
fc_5_auc_df = pd.DataFrame({labels[i]:fc_5_auc[i] for i in range(len(labels))}).melt(value_vars=labels,value_name='AUC',var_name='decoder')
fc_5_auc_df['encoder'] = 'fc5'



filename = 'test_corrs.pkl'
#fc_3_corr = [pd.read_pickle(fc_3_g_3+filename),pd.read_pickle(fc_3_g_4+filename),pd.read_pickle(fc_3_g_5+filename)]
fc_3_corr = [pd.read_pickle(fc_3_g_3+filename),pd.read_pickle(fc_3_g_5_4+filename),pd.read_pickle(fc_3_g_4+filename),pd.read_pickle(fc_3_g_5_5+filename),pd.read_pickle(fc_3_g_5+filename)]
fc_3_corr.reverse()
fc_3_corr_df = pd.DataFrame({labels[i]:fc_3_corr[i] for i in range(len(labels))}).melt(value_vars=labels,value_name='Corr',var_name='decoder')
fc_3_corr_df['encoder'] = 'fc3'

fc_5_4_corr = [pd.read_pickle(fc_5_4_g_3+filename),pd.read_pickle(fc_5_4_g_5_4+filename),pd.read_pickle(fc_5_4_g_4+filename),pd.read_pickle(fc_5_4_g_5_5+filename),pd.read_pickle(fc_5_4_g_5+filename)]
fc_5_4_corr.reverse()
fc_5_4_corr_df = pd.DataFrame({labels[i]:fc_5_4_corr[i] for i in range(len(labels))}).melt(value_vars=labels,value_name='Corr',var_name='decoder')
fc_5_4_corr_df['encoder'] = 'fc54'

fc_4_corr = [pd.read_pickle(fc_4_g_3+filename),pd.read_pickle(fc_4_g_5_4+filename),pd.read_pickle(fc_4_g_4+filename),pd.read_pickle(fc_3_g_5_5+filename),pd.read_pickle(fc_4_g_5+filename)]
fc_4_corr.reverse()
fc_4_corr_df = pd.DataFrame({labels[i]:fc_4_corr[i] for i in range(len(labels))}).melt(value_vars=labels,value_name='Corr',var_name='decoder')
fc_4_corr_df['encoder'] = 'fc4'

fc_5_corr = [pd.read_pickle(fc_5_g_3+filename),pd.read_pickle(fc_5_g_5_4+filename),pd.read_pickle(fc_5_g_4+filename),pd.read_pickle(fc_5_g_5_5+filename),pd.read_pickle(fc_5_g_5+filename)]
fc_5_corr.reverse()
fc_5_corr_df = pd.DataFrame({labels[i]:fc_5_corr[i] for i in range(len(labels))}).melt(value_vars=labels,value_name='Corr',var_name='decoder')
fc_5_corr_df['encoder'] = 'fc5'




corr_df = pd.concat([fc_5_corr_df,fc_4_corr_df,fc_5_4_corr_df,fc_3_corr_df],axis=0)
print(corr_df)

auc_df = pd.concat([fc_5_auc_df,fc_4_auc_df,fc_5_4_auc_df,fc_3_auc_df],axis=0)
print(auc_df)

print("correlation")
corr_model = ols('Corr ~ C(encoder) + C(decoder) + C(encoder):C(decoder)',data=corr_df).fit()
print(sm.stats.anova_lm(corr_model,typ=2))

print("0:fc3, 1:fc54, 2:fc4, 3:fc5")
corr_enc_results = tukey_hsd(corr_df[corr_df['encoder']=='fc3']['Corr'],corr_df[corr_df['encoder']=='fc54']['Corr'],corr_df[corr_df['encoder']=='fc4']['Corr'],corr_df[corr_df['encoder']=='fc5']['Corr'])
print(corr_enc_results)

print("0:g3, 1:g54, 2:g4, 3:g55, 4:g5")
corr_dec_results = tukey_hsd(corr_df[corr_df['decoder']=='g3']['Corr'],corr_df[corr_df['decoder']=='g54']['Corr'],corr_df[corr_df['decoder']=='g4']['Corr'],corr_df[corr_df['decoder']=='g5']['Corr'],corr_df[corr_df['decoder']=='g55']['Corr'])
print(corr_dec_results)

print("auc")
auc_model = ols('AUC ~ C(encoder) + C(decoder) + C(encoder):C(decoder)',data=auc_df).fit()
print(sm.stats.anova_lm(auc_model,typ=2))

print("0:fc3, 1:fc54, 2:fc4, 3:fc5")
auc_enc_results = tukey_hsd(auc_df[auc_df['encoder']=='fc3']['AUC'],auc_df[auc_df['encoder']=='fc54']['AUC'],auc_df[auc_df['encoder']=='fc4']['AUC'],auc_df[auc_df['encoder']=='fc5']['AUC'])
print(auc_enc_results)

print("0:g3, 1:g54, 2:g4, 3:g55, 4:g5")
auc_dec_results = tukey_hsd(auc_df[auc_df['decoder']=='g3']['AUC'],auc_df[auc_df['decoder']=='g54']['AUC'],auc_df[auc_df['decoder']=='g4']['AUC'],auc_df[auc_df['decoder']=='g5']['AUC'],auc_df[auc_df['decoder']=='g55']['AUC'])
print(auc_dec_results)


l2_corrs = [fc_5_corr,fc_4_corr,fc_5_4_corr,fc_3_corr]
l2_aucs = [fc_5_auc,fc_4_auc,fc_5_4_auc,fc_3_auc]
