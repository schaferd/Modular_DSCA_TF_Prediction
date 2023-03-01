import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import ttest_1samp, ttest_ind
import pandas as pd

base_path = '/nobackup/users/schaferd/ae_project_outputs/reg_tests/'
control = base_path+'control_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_rel_conn10_2-12_11.31.31/'
do_2 = base_path+'do_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_do0.2_rel_conn10_2-13_9.41.14/'
do_4 = base_path+'do_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_do0.4_rel_conn10_2-13_9.41.13/'
do_6 = base_path+'do_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_do0.6_rel_conn10_2-13_9.41.12/'
do_8 = base_path+'do_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_do0.8_rel_conn10_2-13_9.41.12/'
do_1 = base_path+'do_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_do1.0_rel_conn10_2-13_12.39.1/'


control_auc = pd.read_pickle(control+'aucs.pkl')
do_2_auc = pd.read_pickle(do_2+'aucs.pkl')
do_4_auc = pd.read_pickle(do_4+'aucs.pkl')
do_6_auc = pd.read_pickle(do_6+'aucs.pkl')
do_8_auc = pd.read_pickle(do_8+'aucs.pkl')
do_1_auc = pd.read_pickle(do_1+'aucs.pkl') 


##CORRS
control_corr = pd.read_pickle(control+'test_corrs.pkl')
do_2_corr =  pd.read_pickle(do_2+'test_corrs.pkl')
do_4_corr = pd.read_pickle(do_4+'test_corrs.pkl')
do_6_corr = pd.read_pickle(do_6+'test_corrs.pkl')
do_8_corr = pd.read_pickle(do_8+'test_corrs.pkl')
do_1_corr = pd.read_pickle(do_1+'test_corrs.pkl') 

do_corrs = [control_corr,do_2_corr,do_4_corr,do_6_corr,do_8_corr,do_1_corr]
do_aucs = [control_auc,do_2_auc,do_4_auc,do_6_auc,do_8_auc,do_1_auc]

x = [0,0.2,0.4,0.6,0.8,1]
do_corr_errors = []
do_auc_errors = []
do_corr_mean = []
do_auc_mean = []

for i in range(len(do_aucs)):
    do_corr_errors.append(np.std(np.array(do_corrs[i])))
    do_auc_errors.append(np.std(np.array(do_aucs[i])))
    do_corr_mean.append(np.mean(np.array(do_corrs[i])))
    do_auc_mean.append(np.mean(np.array(do_aucs[i])))


labels = []
pvals = []
for i in range(len(do_aucs)):
    for j in range(i+1,len(do_aucs)):
        stat,pval = ttest_ind(do_aucs[i],do_aucs[j])
        pvals.append(pval)
        labels.append(str(x[i])+', '+str(x[j]))

p_adjusted = multipletests(pvals,alpha=0.05,method='bonferroni')[1]
for i in range(len(p_adjusted)):
    if p_adjusted[i] < 0.05:
        print(labels[i],p_adjusted[i])
        
#CORRELATION
labels = []
pvals = []
for i in range(len(do_corrs)):
    for j in range(i+1,len(do_corrs)):
        stat,pval = ttest_ind(do_corrs[i],do_corrs[j])
        pvals.append(pval)
        labels.append(str(x[i])+', '+str(x[j]))

print("CORRELATION")
p_adjusted = multipletests(pvals,alpha=0.05,method='bonferroni')[1]
for i in range(len(p_adjusted)):
    if p_adjusted[i] < 0.05:
        print(labels[i],p_adjusted[i])
        

"""
fig, ax = plt.subplots()

print(do_auc_mean)
print(do_auc_errors)
ax.errorbar(x,do_auc_mean,yerr = do_auc_errors,ecolor='k',alpha=0.5, capsize=4)
ax.errorbar(x,do_corr_mean,yerr=do_corr_errors,ecolor='k',alpha=0.5,capsize=4)
fig.savefig('do_tests.png')

"""

