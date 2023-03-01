import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from split_l2_data import l2_corrs, l2_aucs

fig, ax = plt.subplots()
fig.set_figwidth(12)
fig.set_figheight(5.5)
subfigs = fig.subfigures(2,1)
plt.subplots_adjust(left=0.2,bottom=0.2,right=0.8,top=0.75,wspace=0.2,hspace=2)

x_ticks = [[1e-5,5e-5,1e-4,5e-4,1e-3],[1e-5,5e-5,1e-4,5e-4,1e-3],[1e-5,5e-5,1e-4,5e-4,1e-3],[1e-5,1e-4,1e-3]]
x_ticks.reverse()
titles = ['Encoder L2 1e-5','Encoder L2 1e-4','Encoder L2 5e-4','Encoder L2 1e-3']



#CORR
axBottom=subfigs[0].subplots(1,4,sharey=True,sharex=True)
for i, a in enumerate(axBottom):
    l2_corrs_mean = np.array(l2_corrs[i]).mean(axis=1)
    l2_corrs_std = np.array(l2_corrs[i]).std(axis=1)
    print(len(x_ticks[i]),len(l2_corrs_mean))
    a.errorbar(x_ticks[i],l2_corrs_mean,yerr=l2_corrs_std,ecolor='k',alpha=0.5,capsize=4)
    a.set_xscale('log')
    a.set_title(titles[i])
    a.set_xlabel('Decoder L2')
    if i == 0:
        a.set_ylabel('Correlation')
    else:
        a.set_ylabel('')
subfigs[0].suptitle('F-G L2 Regularization vs. Reconstruction Correlation',fontsize='x-large',y=0.95)

#AUC ROC
axBottom=subfigs[1].subplots(1,4,sharey=True,sharex=True)
for i, a in enumerate(axBottom):
    l2_auc_mean = np.array(l2_aucs[i]).mean(axis=1)
    l2_auc_std = np.array(l2_aucs[i]).std(axis=1)
    a.errorbar(x_ticks[i],l2_auc_mean,yerr=l2_auc_std,ecolor='k',alpha=0.5,capsize=4)
    a.set_xscale('log')
    a.set_title(titles[i])
    a.set_xlabel('Decoder L2')
    if i == 0:
        a.set_ylabel('ROC AUC')
    else:
        a.set_ylabel('')
subfigs[1].suptitle('F-G L2 Regularization vs. Perturbation Performance',fontsize='x-large',y=0.95)
fig.savefig('split_l2_plots.png', bbox_inches='tight')

