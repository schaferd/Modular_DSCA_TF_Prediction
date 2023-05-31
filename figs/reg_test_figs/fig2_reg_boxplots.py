import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


from do_data import do_corr_errors, do_auc_errors, do_corr_mean, do_auc_mean,do_aucs,do_corrs
from noise_data import noise_corr_errors, noise_auc_errors, noise_corr_mean, noise_auc_mean,noise_aucs,noise_corrs
from l2_data import l2_corr_errors, l2_auc_errors, l2_corr_mean, l2_auc_mean, l2_aucs,l2_corrs

TITLE_Y = 1 

PROPS = {
    #    'boxprops':{'alpha':0.6},
    'boxprops':{'facecolor':'white', 'edgecolor':'gray'},
    'medianprops':{'color':'gray'},
    'whiskerprops':{'color':'gray'},
    'capprops':{'color':'gray'}
}

fig,ax = plt.subplots()
subfigs = fig.subfigures(3,1)
#plt.tight_layout()
plt.subplots_adjust(left=0.2,bottom=0.35,right=0.8,top=1,wspace=0.34,hspace=0.8)
fig.set_figwidth(9)
fig.set_figheight(9.5)

#L2
axTop = subfigs[0].subplots(1,2)
#x = [0.1,0.01,0.001,0.0001,0]
x = [0,0.00001,0.0001,0.001,0.01]

a = axTop[0]
sns.boxplot(data=l2_corrs,ax=a,**PROPS)
sns.swarmplot(data=l2_corrs,ax=a,edgecolor='gray',linewidth=1,alpha=0.8)
a.set_xlabel('L2 Constant')
a.set_xticklabels(x)
a.set_ylim(0, 1)
a.set_ylabel('Correlation')
a.set_title('L2 Constant vs. \nReconstruction Correlation', y=TITLE_Y)

a = axTop[1]
a.set_ylabel('ROC AUC')
a.axhline(y=0.5, color='darkgrey', linestyle='--',zorder=0)
sns.boxplot(data=l2_aucs,ax=a,**PROPS)
sns.swarmplot(data=l2_aucs,ax=a,edgecolor='gray',linewidth=1,alpha=0.8)
a.set_ylim(0.4, 0.8)
a.set_xticklabels(x)
a.set_xlabel('L2 Constant')
a.set_title('L2 Constant vs. \nPerturbation Validation',y=TITLE_Y)

#NOISE
axMiddle = subfigs[1].subplots(1,2,sharex=True)
x = [0,0.001,0.01,0.1,1]
a = axMiddle[0]
sns.boxplot(data=noise_corrs,ax=a,**PROPS)
sns.swarmplot(data=noise_corrs,ax=a,edgecolor='gray',linewidth=1,alpha=0.8)
a.set_xlabel('Noise Order of Magnitude')
a.set_xticklabels(x)
a.set_ylim(0, 1)
a.set_ylabel('Correlation')
a.set_title('Noise vs. \nReconstruction Correlation',y=TITLE_Y)

a = axMiddle[1]
sns.boxplot(data=noise_aucs,ax=a,**PROPS)
sns.swarmplot(data=noise_aucs,ax=a,edgecolor='gray',linewidth=1,alpha=0.8)
a.set_ylabel('ROC AUC')
a.axhline(y=0.5, color='darkgrey', linestyle='--',zorder=0)
a.set_xticklabels(x)
a.set_ylim(0.4, 0.8)
a.set_xlabel('Noise Order of Magnitude')
a.set_title('Noise vs. \nPerturbation Validation',y=TITLE_Y)

#DROPOUT
axBottom = subfigs[2].subplots(1,2,sharex=True)
x = [0,0.2,0.4,0.6,0.8,1]
a = axBottom[0]
sns.boxplot(data=do_corrs,ax=a,**PROPS)
sns.swarmplot(data=do_corrs,ax=a,edgecolor='gray',linewidth=1,alpha=0.8)
a.set_xlabel('Dropout Rate')
a.set_ylim(0, 1)
a.set_xticklabels(x)
a.set_ylabel('Correlation')
a.set_title('Dropout Rate vs. \nReconstruction Correlation',y=TITLE_Y)

a = axBottom[1]
sns.boxplot(data=do_aucs,ax=a,**PROPS)
sns.swarmplot(data=do_aucs,ax=a,edgecolor='gray',linewidth=1,alpha=0.8)
a.set_ylabel('ROC AUC')
a.set_ylim(0.4, 0.8)
a.set_xlabel('Dropout Rate')
a.set_xticklabels(x)
a.axhline(y=0.5, color='darkgrey', linestyle='--',zorder=0)
a.set_title('Dropout Rate vs. \nPerturbation Validation',y=TITLE_Y)


fig.suptitle('F-G Regularization Search',fontsize='x-large',y=0.95)

fig.savefig('reg_plots.png', bbox_inches='tight',dpi=300)
