import numpy as np
from scipy.stats import ttest_1samp, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

base_path='/nobackup/users/schaferd/ae_project_outputs/moa_tests/'
moa_dir = base_path+'moa_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_rel_conn10_2-16_16.26.21/'
no_moa_dir = base_path+'no_moa_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_rel_conn10_2-16_16.26.54/'

moa_l2_dir = base_path+'moa_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_2-16_8.29.4'
no_moa_l2_dir = base_path+'no_moa_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_rel_conn10_2-16_8.29.16'

moa_auc = pd.read_pickle(moa_dir+'aucs.pkl')
no_moa_auc = pd.read_pickle(no_moa_dir+'aucs.pkl')

moa_corr = pd.read_pickle(moa_dir+'test_corrs.pkl')
no_moa_corr = pd.read_pickle(no_moa_dir+'test_corrs.pkl')

fig,ax = plt.subplots(1,2)
plt.subplots_adjust(left=0.2,bottom=0.2,right=0.8,top=0.8,wspace=0.4,hspace=2)
fig.set_figwidth(8)
fig.set_figheight(4)

corr_data = [no_moa_corr,moa_corr]
auc_data = [no_moa_auc,moa_auc]

#CORRELATION
a = ax[0]
with sns.color_palette("Paired"):
    sns.boxplot(data=corr_data,ax=a,boxprops=dict(alpha=0.3),showfliers=False)
    sns.swarmplot(data=corr_data,ax=a, edgecolor='k',linewidth=1)
    a.set_xticklabels(["No MOA","MOA"])
    a.set_ylim(0,1)
    a.set_ylabel("Correlation")
    a.set_title('Reconstruction Correlation',y=1.01)


#AUC
a = ax[1]
with sns.color_palette("Paired"):
    sns.boxplot(data=auc_data,ax=a,boxprops=dict(alpha=0.3),showfliers=False)
    sns.swarmplot(data=auc_data,ax=a,edgecolor='k',linewidth=1)
    a.set_xticklabels(["No MOA","MOA"])
    a.set_ylabel("ROC AUC")
    a.axhline(y=0.5, color='r', linestyle='--',alpha=0.4)
    a.set_ylim(0.4,0.8)
    a.set_title('TF Knockout Prediction',y=1.01)

fig.suptitle('F-G MOA vs. No MOA Performance', fontsize='x-large',y=0.93)
fig.savefig('moa_boxplots.png', bbox_inches='tight')

plt.clf()

fig,ax = plt.subplots(1,2)
