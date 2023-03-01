import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd


from do_data import do_corr_errors, do_auc_errors, do_corr_mean, do_auc_mean
from noise_data import noise_corr_errors, noise_auc_errors, noise_corr_mean, noise_auc_mean
from l2_data import l2_corr_errors, l2_auc_errors, l2_corr_mean, l2_auc_mean

TITLE_Y = 1 

fig,ax = plt.subplots()
subfigs = fig.subfigures(3,1)
#plt.tight_layout()
plt.subplots_adjust(left=0.2,bottom=0.35,right=0.8,top=1,wspace=0.34,hspace=0.8)
fig.set_figwidth(9)
fig.set_figheight(9.5)

#L2
axTop = subfigs[0].subplots(1,2)
#x = [0.1,0.01,0.001,0.0001,0]
x = [0,0.00001,0.0001,0.001,0.01,0.1]

a = axTop[0]
a.set_xscale('log')
a.errorbar(x,l2_corr_mean,yerr=l2_corr_errors,ecolor='k',alpha=0.5,capsize=4)
a.set_xlabel('L2 Constant')
a.set_ylim(0, 0.7)
a.set_ylabel('Correlation')
a.set_title('L2 Constant vs. \nReconstruction Correlation', y=TITLE_Y)

a = axTop[1]
a.errorbar(x,l2_auc_mean,yerr = l2_auc_errors,ecolor='k',alpha=0.5, capsize=4)
a.set_ylabel('ROC AUC')
a.axhline(y=0.5, color='r', linestyle='--',alpha=0.4)
a.set_ylim(0.4, 0.8)
a.set_xscale('log')
a.set_xlabel('L2 Constant')
a.set_title('L2 Constant vs. \nTF Knock Out Prediction',y=TITLE_Y)

#NOISE
axMiddle = subfigs[1].subplots(1,2,sharex=True)
x = [0,0.001,0.01,0.1,1]
a = axMiddle[0]
a.set_xscale('log')
a.errorbar(x,noise_corr_mean,yerr=noise_corr_errors,ecolor='k',alpha=0.5,capsize=4)
a.set_xlabel('Noise Order of Magnitude')
a.set_ylim(0, 0.7)
a.set_ylabel('Correlation')
a.set_title('Noise vs. \nReconstruction Correlation',y=TITLE_Y)

a = axMiddle[1]
a.errorbar(x,noise_auc_mean,yerr = noise_auc_errors,ecolor='k',alpha=0.5, capsize=4)
a.set_ylabel('ROC AUC')
a.axhline(y=0.5, color='r', linestyle='--',alpha=0.4)
a.set_ylim(0.4, 0.8)
a.set_xscale('log')
a.set_xlabel('Noise Order of Magnitude')
a.set_title('Noise vs. \nTF Knock Out Prediction',y=TITLE_Y)

#DROPOUT
axBottom = subfigs[2].subplots(1,2,sharex=True)
x = [0,0.2,0.4,0.6,0.8,1]
a = axBottom[0]
a.errorbar(x,do_corr_mean,yerr=do_corr_errors,ecolor='k',alpha=0.5,capsize=4)
a.set_xlabel('Dropout Rate')
a.set_ylim(0, 0.7)
a.set_ylabel('Correlation')
a.set_title('Dropout Rate vs. \nReconstruction Correlation',y=TITLE_Y)

a = axBottom[1]
a.errorbar(x,do_auc_mean,yerr = do_auc_errors,ecolor='k',alpha=0.5, capsize=4)
a.set_ylabel('ROC AUC')
a.set_ylim(0.4, 0.8)
a.set_xlabel('Dropout Rate')
a.axhline(y=0.5, color='r', linestyle='--',alpha=0.4)
a.set_title('Dropout Rate vs. \nTF Knock Out Prediction',y=TITLE_Y)


fig.suptitle('F-G Regularization Search',fontsize='x-large',y=0.95)

fig.savefig('reg_plots.png', bbox_inches='tight')
