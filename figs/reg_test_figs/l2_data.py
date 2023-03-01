import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import ttest_1samp, ttest_ind
import pandas as pd

base_path = '/nobackup/users/schaferd/ae_project_outputs/reg_tests/'
control = base_path+'control_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_rel_conn10_2-12_11.31.31/'
l2_5_1 = base_path+'l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del21e-05_enl21e-05_moa1.0_rel_conn10_2-12_11.31.31/'
l2_5_2 = base_path+'reg_tests_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del21e-05_enl21e-05_moa1.0_rel_conn10_2-26_12.16.52/'
l2_4_1 = base_path+'l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0001_moa1.0_rel_conn10_2-12_11.31.35/'
l2_4_2 = base_path+'reg_tests_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0001_moa1.0_rel_conn10_2-26_12.16.52/'
l2_3 = base_path+'l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.001_enl20.001_moa1.0_rel_conn10_2-12_11.31.45/'
l2_2 = base_path+'l2_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_2-12_14.31.34/'

control_auc = pd.read_pickle(control+'aucs.pkl')
l2_1_auc = [] #pd.read_pickle(l2_1+'aucs.pkl')
l2_2_auc = pd.read_pickle(l2_2+'aucs.pkl')
l2_3_auc = pd.read_pickle(l2_3+'aucs.pkl')
l2_4_auc = pd.read_pickle(l2_4_1+'aucs.pkl')+pd.read_pickle(l2_4_2+'aucs.pkl')
l2_5_auc = pd.read_pickle(l2_5_1+'aucs.pkl')+pd.read_pickle(l2_5_2+'aucs.pkl')

control_corr = pd.read_pickle(control+'test_corrs.pkl')
l2_1_corr = [] #pd.read_pickle(l2_1+'test_corrs.pkl')
l2_2_corr = pd.read_pickle(l2_2+'test_corrs.pkl')
l2_3_corr = pd.read_pickle(l2_3+'test_corrs.pkl')
l2_4_corr = pd.read_pickle(l2_4_1+'test_corrs.pkl')+pd.read_pickle(l2_4_2+'test_corrs.pkl')
l2_5_corr = pd.read_pickle(l2_5_1+'test_corrs.pkl')+pd.read_pickle(l2_5_2+'test_corrs.pkl')

l2_corrs = [control_corr,l2_5_corr,l2_4_corr,l2_3_corr,l2_2_corr,l2_1_corr]
l2_aucs = [control_auc,l2_5_auc, l2_4_auc,l2_3_auc,l2_2_auc,l2_1_auc]

l2_corr_errors = []
l2_auc_errors = []
l2_corr_mean = []
l2_auc_mean = []

for i in range(len(l2_aucs)):
    l2_corr_errors.append(np.std(np.array(l2_corrs[i])))
    l2_auc_errors.append(np.std(np.array(l2_aucs[i])))
    l2_corr_mean.append(np.mean(np.array(l2_corrs[i])))
    l2_auc_mean.append(np.mean(np.array(l2_aucs[i])))

x = [0,-5,-4,-3,-2,-2,-1]
labels = []
pvals = []
for i in range(len(l2_aucs)):
    for j in range(i+1,len(l2_aucs)):
        stat,pval = ttest_ind(l2_aucs[i],l2_aucs[j])
        pvals.append(pval)
        labels.append(str(x[i])+', '+str(x[j]))

p_adjusted = multipletests(pvals,alpha=0.05,method='bonferroni')[1]
for i in range(len(p_adjusted)):
    #if p_adjusted[i] < 0.05:
    print(labels[i],p_adjusted[i])
