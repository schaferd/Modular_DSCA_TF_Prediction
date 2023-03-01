import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import ttest_1samp, ttest_ind
import pandas as pd

base_path = '/nobackup/users/schaferd/ae_project_outputs/reg_tests/'
control = base_path+'control_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_rel_conn10_2-12_11.31.31/'

noise_001 = base_path+'noise_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_noise0.01_rel_conn10_2-13_13.37.37/' #noise = 0.001
noise_01 = base_path+'noise_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_noise0.1_rel_conn10_2-13_13.38.8/' #noise = 0.01
noise_1 = base_path+'noise_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_noise1.0_rel_conn10_2-13_13.38.8/' #noise = 0.1
noise_10 = base_path+'noise_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_noise10.0_rel_conn10_2-13_15.44.29/' #noise = 1
noise_100 =base_path+'noise_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_noise100.0_rel_conn10_2-13_17.17.33/'  #noise = 10

control_auc = pd.read_pickle(control+'aucs.pkl')
noise_001_auc = pd.read_pickle(noise_001+'aucs.pkl') #noise = 0.001
noise_01_auc = pd.read_pickle(noise_01+'aucs.pkl') #noise = 0.01
noise_1_auc = pd.read_pickle(noise_1+'aucs.pkl') #noise = 0.1
noise_10_auc = pd.read_pickle(noise_10+'aucs.pkl') #noise = 1
noise_100_auc = pd.read_pickle(noise_100+'aucs.pkl') #noise = 10


control_corr = pd.read_pickle(control+'test_corrs.pkl')
noise_001_corr = pd.read_pickle(noise_001+'test_corrs.pkl') #noise = 0.001
noise_01_corr = pd.read_pickle(noise_01+'test_corrs.pkl') #noise = 0.01
noise_1_corr = pd.read_pickle(noise_1+'test_corrs.pkl') #noise = 0.1
noise_10_corr = pd.read_pickle(noise_10+'test_corrs.pkl') #noise = 1
noise_100_corr = pd.read_pickle(noise_100+'test_corrs.pkl') #noise = 10


noise_corrs = [control_corr,noise_001_corr,noise_01_corr,noise_1_corr,noise_10_corr]#,noise_100_corr]
noise_aucs = [control_auc,noise_001_auc,noise_01_auc,noise_1_auc,noise_10_auc]#,noise_100_auc]

noise_corr_errors = []
noise_auc_errors = []
noise_corr_mean = []
noise_auc_mean = []


for i in range(len(noise_aucs)):
    noise_corr_errors.append(np.std(np.array(noise_corrs[i])))
    noise_auc_errors.append(np.std(np.array(noise_aucs[i])))
    noise_corr_mean.append(np.mean(np.array(noise_corrs[i])))
    noise_auc_mean.append(np.mean(np.array(noise_aucs[i])))


labels = []
pvals = []
x = [0,0.001,0.01,0.1,1,10]
for i in range(len(noise_aucs)):
    for j in range(i+1,len(noise_aucs)):
        stat,pval = ttest_ind(noise_aucs[i],noise_aucs[j])
        pvals.append(pval)
        labels.append(str(x[i])+', '+str(x[j]))

p_adjusted = multipletests(pvals,alpha=0.05,method='bonferroni')[1]
for i in range(len(p_adjusted)):
    if p_adjusted[i] < 0.05:
        print(labels[i],p_adjusted[i])

labels = []
pvals = []
print("CORRELATION")
for i in range(len(noise_corrs)):
    for j in range(i+1,len(noise_corrs)):
        stat,pval = ttest_ind(noise_corrs[i],noise_corrs[j])
        pvals.append(pval)
        labels.append(str(x[i])+', '+str(x[j]))

p_adjusted = multipletests(pvals,alpha=0.05,method='bonferroni')[1]
for i in range(len(p_adjusted)):
    if p_adjusted[i] < 0.05:
        print(labels[i],p_adjusted[i])
