import pandas as pd
import numpy as np

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
fc_5_4_g_1_5 = base_path+'split_l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del21e-05_enl20.0005_moa1.0_rel_conn10_2-15_21.39.33/'

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


filename = 'aucs.pkl'
fc_3_auc = [pd.read_pickle(fc_3_g_3+filename),pd.read_pickle(fc_3_g_5_4+filename),pd.read_pickle(fc_3_g_4+filename),pd.read_pickle(fc_3_g_5_5+filename),pd.read_pickle(fc_3_g_5+filename)]
fc_3_auc.reverse()
#fc_4_auc = [pd.read_pickle(fc_4_g_3+filename),pd.read_pickle(fc_4_g_4+filename),pd.read_pickle(fc_4_g_5+filename)]
fc_5_4_auc = [pd.read_pickle(fc_5_4_g_3+filename),pd.read_pickle(fc_5_4_g_5_4+filename),pd.read_pickle(fc_5_4_g_4+filename),pd.read_pickle(fc_5_4_g_5_5+filename),pd.read_pickle(fc_5_4_g_1_5+filename)]
fc_5_4_auc.reverse()
fc_4_auc= [pd.read_pickle(fc_4_g_3+filename),pd.read_pickle(fc_4_g_5_4+filename),pd.read_pickle(fc_4_g_4+filename),pd.read_pickle(fc_3_g_5_5+filename),pd.read_pickle(fc_4_g_5+filename)]
fc_4_auc.reverse()
fc_5_auc = [pd.read_pickle(fc_5_g_3+filename),pd.read_pickle(fc_5_g_5_4+filename),pd.read_pickle(fc_5_g_4+filename),pd.read_pickle(fc_5_g_5_5+filename),pd.read_pickle(fc_5_g_5+filename)]
fc_5_auc.reverse()

filename = 'test_corrs.pkl'
#fc_3_corr = [pd.read_pickle(fc_3_g_3+filename),pd.read_pickle(fc_3_g_4+filename),pd.read_pickle(fc_3_g_5+filename)]
fc_3_corr = [pd.read_pickle(fc_3_g_3+filename),pd.read_pickle(fc_3_g_5_4+filename),pd.read_pickle(fc_3_g_4+filename),pd.read_pickle(fc_3_g_5_5+filename),pd.read_pickle(fc_3_g_5+filename)]
fc_3_corr.reverse()
fc_5_4_corr = [pd.read_pickle(fc_5_4_g_3+filename),pd.read_pickle(fc_5_4_g_5_4+filename),pd.read_pickle(fc_5_4_g_4+filename),pd.read_pickle(fc_5_4_g_5_5+filename),pd.read_pickle(fc_5_4_g_1_5+filename)]
fc_5_4_corr.reverse()
fc_4_corr = [pd.read_pickle(fc_4_g_3+filename),pd.read_pickle(fc_4_g_5_4+filename),pd.read_pickle(fc_4_g_4+filename),pd.read_pickle(fc_3_g_5_5+filename),pd.read_pickle(fc_4_g_5+filename)]
fc_4_corr.reverse()
fc_5_corr = [pd.read_pickle(fc_5_g_3+filename),pd.read_pickle(fc_5_g_5_4+filename),pd.read_pickle(fc_5_g_4+filename),pd.read_pickle(fc_5_g_5_5+filename),pd.read_pickle(fc_5_g_5+filename)]
fc_5_corr.reverse()

l2_corrs = [fc_5_corr,fc_4_corr,fc_5_4_corr,fc_3_corr]
l2_aucs = [fc_5_auc,fc_4_auc,fc_5_4_auc,fc_3_auc]
