import pandas as pd
import numpy as np

base_path = '/nobackup/users/schaferd/ae_project_outputs/reg_tests/'

fc_2_g_2 = base_path+'shallow_split_l2_shallow-shallow_epochs50_batchsize128_enlr0.0001_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_2-15_21.53.59/'
fc_2_g_3 = base_path+'shallow_split_l2_shallow-shallow_epochs50_batchsize128_enlr0.0001_delr0.01_del20.001_enl20.01_moa1.0_rel_conn10_2-15_21.53.58/'
fc_2_g_4 = base_path+'shallow_split_l2_shallow-shallow_epochs50_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.01_moa1.0_rel_conn10_2-15_23.14.41/'

fc_3_g_3 = base_path+'shallow_l2_shallow-shallow_epochs50_batchsize128_enlr0.0001_delr0.01_del20.001_enl20.001_moa1.0_rel_conn10_2-15_17.36.42/'
fc_3_g_4 = base_path+'shallow_l2_shallow-shallow_epochs50_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.001_moa1.0_rel_conn10_2-15_17.36.42/'
fc_3_g_5 = base_path+'shallow_l2_shallow-shallow_epochs50_batchsize128_enlr0.0001_delr0.01_del21e-05_enl20.001_moa1.0_rel_conn10_2-15_17.36.42/'

fc_4_g_3 = base_path+'shallow_l2_shallow-shallow_epochs50_batchsize128_enlr0.0001_delr0.01_del20.001_enl20.0001_moa1.0_rel_conn10_2-15_17.36.45/'
fc_4_g_4 = base_path+'shallow_l2_shallow-shallow_epochs50_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0001_moa1.0_rel_conn10_2-15_19.0.3/'
fc_4_g_5 = base_path+'shallow_l2_shallow-shallow_epochs50_batchsize128_enlr0.0001_delr0.01_del21e-05_enl20.0001_moa1.0_rel_conn10_2-15_19.58.2/'

fc_5_g_3 = base_path+'shallow_l2_shallow-shallow_epochs50_batchsize128_enlr0.0001_delr0.01_del20.001_enl21e-05_moa1.0_rel_conn10_2-15_19.58.2/'
fc_5_g_4 = base_path+'shallow_l2_shallow-shallow_epochs50_batchsize128_enlr0.0001_delr0.01_del20.0001_enl21e-05_moa1.0_rel_conn10_2-15_19.58.7/'
fc_5_g_5 = base_path+'shallow_l2_shallow-shallow_epochs50_batchsize128_enlr0.0001_delr0.01_del21e-05_enl21e-05_moa1.0_rel_conn10_2-15_21.37.10/'


filename = 'aucs.pkl'
fc_5_auc = [pd.read_pickle(fc_5_g_3+filename),pd.read_pickle(fc_5_g_4+filename),pd.read_pickle(fc_5_g_5+filename)]
fc_5_auc.reverse()
fc_4_auc = [pd.read_pickle(fc_4_g_3+filename),pd.read_pickle(fc_4_g_4+filename),pd.read_pickle(fc_4_g_5+filename)]
fc_4_auc.reverse()
fc_3_auc = [pd.read_pickle(fc_3_g_3+filename),pd.read_pickle(fc_3_g_4+filename),pd.read_pickle(fc_3_g_5+filename)]
fc_3_auc.reverse()
fc_2_auc = [pd.read_pickle(fc_2_g_2+filename),pd.read_pickle(fc_2_g_3+filename),pd.read_pickle(fc_2_g_4+filename)]
fc_2_auc.reverse()

filename = 'test_corrs.pkl'
fc_5_corr = [pd.read_pickle(fc_5_g_3+filename),pd.read_pickle(fc_5_g_4+filename),pd.read_pickle(fc_5_g_5+filename)]
fc_5_corr.reverse()
fc_4_corr = [pd.read_pickle(fc_4_g_3+filename),pd.read_pickle(fc_4_g_4+filename),pd.read_pickle(fc_4_g_5+filename)]
fc_4_corr.reverse()
fc_3_corr = [pd.read_pickle(fc_3_g_3+filename),pd.read_pickle(fc_3_g_4+filename),pd.read_pickle(fc_3_g_5+filename)]
fc_3_corr.reverse()
fc_2_corr = [pd.read_pickle(fc_2_g_2+filename),pd.read_pickle(fc_2_g_3+filename),pd.read_pickle(fc_2_g_4+filename)]
fc_2_corr.reverse()

l2_corrs = [fc_5_corr,fc_4_corr,fc_3_corr,fc_2_corr]
l2_aucs = [fc_5_auc,fc_4_auc,fc_3_auc,fc_2_auc]
