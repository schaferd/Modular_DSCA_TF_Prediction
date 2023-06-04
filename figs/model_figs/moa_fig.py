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


"""
fig,ax = plt.subplots(1,2)
plt.subplots_adjust(left=0.2,bottom=0.2,right=0.8,top=0.8,wspace=0.4,hspace=2)
fig.set_figwidth(8)
fig.set_figheight(4)
"""

"""
SMALL_SIZE = 14 
MEDIUM_SIZE = 16 
BIGGER_SIZE = 18
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
"""

corr_data = [no_moa_corr,moa_corr]
auc_data = [no_moa_auc,moa_auc]

PROPS = {
    #    'boxprops':{'alpha':0.6},
    'boxprops':{'facecolor':'white', 'edgecolor':'gray'},
    'medianprops':{'color':'gray'},
    'whiskerprops':{'color':'gray'},
    'capprops':{'color':'gray'}
}

def create_moa_fig(fig,label_font_size,title_font_size, subtitle_font_size):
    ax = fig.subplots(1,2)
    fig.subplots_adjust(left=0.2,bottom=0.2,right=0.8,top=0.8,wspace=0.5,hspace=0.2)
    #CORRELATION
    print(ax)
    a = ax[0]
    with sns.color_palette("Paired"):
        sns.boxplot(data=corr_data,ax=a,showfliers=False,**PROPS)
        sns.swarmplot(data=corr_data,ax=a, edgecolor='k',linewidth=1)
        a.set_xticklabels(["No MOA","MOA"])
        a.set_ylim(0,1)
        a.set_ylabel("Correlation")
        a.set_title('Reconstruction Correlation',y=1.05)


    #AUC
    a = ax[1]
    with sns.color_palette("Paired"):
        sns.boxplot(data=auc_data,ax=a,showfliers=False,**PROPS)
        sns.swarmplot(data=auc_data,ax=a,edgecolor='k',linewidth=1)
        a.set_xticklabels(["No MOA","MOA"])
        a.set_ylabel("ROC AUC")
        a.axhline(y=0.5, color='darkgrey', linestyle='--')
        a.set_ylim(0.4,0.8)
        a.set_title('TF Knockout Prediction',y=1.05)

    fig.suptitle('FC-G MOA vs. No MOA Performance', fontsize='x-large',y=0.99)
#fig.savefig('moa_boxplots.png', bbox_inches='tight',dpi=300)

#plt.clf()

#fig,ax = plt.subplots(1,2)


stat, pval = ttest_ind(moa_auc,no_moa_auc)
print("moa vs. no moa",pval)

stat, pval = ttest_1samp(no_moa_auc,0.5)
print("no moa vs. random",pval)
