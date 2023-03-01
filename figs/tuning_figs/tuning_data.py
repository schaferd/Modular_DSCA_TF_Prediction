import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

base_path = '/nobackup/users/schaferd/ae_project_outputs/final_tuning/'

lr_sched = base_path+'tuning_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.0001_lrensched_maxlr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_2-17_9.42.51/'
orig = base_path+'orig_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_2-17_9.42.33/'
lower_lr = base_path+'tuning_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.001_del20.0001_enl20.0005_moa1.0_rel_conn10_2-17_9.42.33/'
lower_l2 = base_path+'tuning_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.001_del20.0001_enl20.0001_moa1.0_rel_conn10_2-17_15.26.44/'

filename = 'aucs.pkl'
auc_data = [pd.read_pickle(orig+filename),pd.read_pickle(lower_l2+filename),pd.read_pickle(lower_lr+filename)]
auc_data_mean = np.array(auc_data).mean(axis=1)
auc_data_std = np.array(auc_data).std(axis=1)

filename = 'test_corrs.pkl'
corr_data = [pd.read_pickle(orig+filename),pd.read_pickle(lower_l2+filename),pd.read_pickle(lower_lr+filename)]
corr_data_mean = np.array(corr_data).mean(axis=1)
corr_data_std = np.array(corr_data).std(axis=1)

x_ticks = ['orig','lower_l2','lower_lr']

fig,ax = plt.subplots(1,2)
a = ax[0]
sns.boxplot(data=corr_data,ax=a, boxprops=dict(alpha=.3))
sns.swarmplot(data=corr_data,ax=a, edgecolor='k',linewidth=1)
a.set_title('Reconstruction Correlation')
a.set_xticklabels(x_ticks)

#AUC ROC
a = ax[1]
sns.boxplot(data=auc_data,ax=a, boxprops=dict(alpha=.3))
sns.swarmplot(data=auc_data,ax=a, edgecolor='k',linewidth=1)
a.set_title('TF Knock Out Prediction')
a.set_xticklabels(x_ticks)

fig.savefig('tuning.png',bbox_inches='tight')



