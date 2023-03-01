import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from shallow_split_l2_data import l2_corrs, l2_aucs

fig, ax = plt.subplots()
fig.set_figwidth(9)
fig.set_figheight(5)
subfigs = fig.subfigures(2,1)
plt.subplots_adjust(left=0.2,bottom=0.2,right=0.8,top=0.8,wspace=0.3,hspace=2)

x_ticks = [[1e-5,1e-4,1e-3],[1e-5,1e-4,1e-3],[1e-5,1e-4,1e-3],[1e-4,1e-3,1e-2]]
titles = ['Encoder L2 1e-5','Encoder L2 1e-4','Encoder L2 1e-3','Encoder L2 1e-2']



#CORR
axBottom=subfigs[0].subplots(1,4,sharey=True,sharex=True)
for i, a in enumerate(axBottom):
    l2_corrs_mean = np.array(l2_corrs[i]).mean(axis=1)
    l2_corrs_std = np.array(l2_corrs[i]).std(axis=1)
    a.errorbar(x_ticks[i],l2_corrs_mean,yerr=l2_corrs_std,ecolor='k',alpha=0.5,capsize=4)
    a.set_xscale('log')
    a.set_title(titles[i])
    a.set_xlabel('Decoder L2')
    a.set_ylabel('Correlation')

#AUC ROC
axBottom=subfigs[1].subplots(1,4,sharey=True,sharex=True)
for i, a in enumerate(axBottom):
    l2_auc_mean = np.array(l2_aucs[i]).mean(axis=1)
    l2_auc_std = np.array(l2_aucs[i]).std(axis=1)
    a.errorbar(x_ticks[i],l2_auc_mean,yerr=l2_auc_std,ecolor='k',alpha=0.5,capsize=4)
    a.set_xscale('log')
    a.set_title(titles[i])
    a.set_xlabel('Decoder L2')
    a.set_ylabel('ROC AUC')
fig.savefig('shallow_split_l2_plots.png', bbox_inches='tight')

