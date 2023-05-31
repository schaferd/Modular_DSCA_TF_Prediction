import matplotlib.pyplot as plt
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import ttest_1samp, ttest_ind
import seaborn as sns
import numpy as np
import pandas as pd

PROPS = {
    #    'boxprops':{'alpha':0.6},
    'boxprops':{'facecolor':'white', 'edgecolor':'gray'},
    'medianprops':{'color':'gray'},
    'whiskerprops':{'color':'gray'},
    'capprops':{'color':'gray'}
}

base_path = '/nobackup/users/schaferd/ae_project_outputs/reg_tests/'

fc_1_g_1 = base_path+'shallow_l2_fc-genefc_epochs50_batchsize128_enlr0.01_delr0.01_del20.1_enl20.1_moa1.0_rel_conn10_2-16_21.11.54/'
fc_2_g_2 = base_path+'shallow_l2_shallow-shallow_epochs50_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_2-16_13.3.11/'
fc_3_g_3 = base_path+'shallow_l2_shallow-shallow_epochs50_batchsize128_enlr0.01_delr0.01_del20.001_enl20.001_moa1.0_rel_conn10_2-16_13.3.15/'
fc_4_g_4 = base_path+'shallow_l2_shallow-shallow_epochs50_batchsize128_enlr0.01_delr0.01_del20.0001_enl20.0001_moa1.0_rel_conn10_2-16_14.22.30/'

filename = 'aucs.pkl'
auc_data = [pd.read_pickle(fc_4_g_4+filename),pd.read_pickle(fc_3_g_3+filename),pd.read_pickle(fc_2_g_2+filename),pd.read_pickle(fc_1_g_1+filename)]
auc_data_mean = np.array(auc_data).mean(axis=1)
auc_data_std = np.array(auc_data).std(axis=1)

filename = 'test_corrs.pkl'
corr_data = [pd.read_pickle(fc_4_g_4+filename),pd.read_pickle(fc_3_g_3+filename),pd.read_pickle(fc_2_g_2+filename),pd.read_pickle(fc_1_g_1+filename)]
corr_data_mean = np.array(corr_data).mean(axis=1)
corr_data_std = np.array(corr_data).std(axis=1)


x_ticks = [1e-4,1e-3,1e-2,1e-1]

fig, ax = plt.subplots(1,2)
fig.set_figwidth(10)
fig.set_figheight(4)
plt.subplots_adjust(left=0.2,bottom=0.2,right=0.8,top=0.8,wspace=0.4,hspace=1.5)

a = ax[0]
#a.errorbar(x_ticks,corr_data_mean,yerr=corr_data_std,ecolor='k',alpha=0.5,capsize=4)
sns.boxplot(data=corr_data,ax=a,**PROPS)
sns.swarmplot(data=corr_data,ax=a,edgecolor='gray',linewidth=1,alpha=0.8)
#a.set_xscale('log')
a.set_ylabel('Correlation')
a.set_xticklabels(x_ticks)
a.set_xlabel('L2 Constant')
a.set_title('L2 Constant vs. \nReconstruction Correlation')


a = ax[1]
#a.errorbar(x_ticks,auc_data_mean,yerr=auc_data_std,ecolor='k',alpha=0.5,capsize=4)
#a.set_xscale('log')
a.axhline(y=0.5, color='darkgrey', linestyle='--',zorder=0)
sns.boxplot(data=auc_data,ax=a,**PROPS)
sns.swarmplot(data=auc_data,ax=a,edgecolor='gray',linewidth=1,alpha=0.8)
a.set_ylabel('AUC ROC')
a.set_xlabel('L2 Constant')
a.set_xticklabels(x_ticks)
a.set_title('L2 Constant vs. \nTF Perturbation Evaluation')

fig.suptitle('Shallow L2 Regularization Performance',fontsize='x-large')

fig.savefig('shallow_l2_plots.png', bbox_inches='tight',dpi=300)

labels = []
pvals = []
for i in range(len(auc_data)):
    for j in range(i+1,len(auc_data)):
        stat,pval = ttest_ind(auc_data[i],auc_data[j])
        pvals.append(pval)
        labels.append(str(x_ticks[i])+', '+str(x_ticks[j]))

p_adjusted = multipletests(pvals,alpha=0.05,method='bonferroni')[1]
for i in range(len(p_adjusted)):
    if p_adjusted[i] < 0.05:
        print(labels[i],p_adjusted[i])
