import pandas as pd
import sys
import os
from statsmodels.sandbox.stats.multicomp import multipletests
import scipy.stats
from scipy.stats import ttest_1samp, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

consistency_eval_path = '/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/'
sys.path.insert(1,consistency_eval_path)
from check_consistency_ko import calculate_consistency, make_random_ranks, kendall_w

plt.rcParams.update({'font.size': 13})

#base_path = '/nobackup/users/schaferd/ae_project_outputs/final_eval/'
base_path = '/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/unfiltered/'
shallow1 = base_path+'shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/'
shallow2 = base_path+'shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_14.26.53/'
deep1 = base_path+'fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.56/'
deep2 = base_path+'fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/'

deep_pert = pd.read_pickle(deep1+'aucs.pkl')+pd.read_pickle(deep2+'aucs.pkl')
deep_knocktf = pd.read_pickle(deep1+'knocktf_aucs.pkl')+pd.read_pickle(deep2+'knocktf_aucs.pkl')
shallow_pert = pd.read_pickle(shallow1+'aucs.pkl')+pd.read_pickle(shallow2+'aucs.pkl')
shallow_knocktf = pd.read_pickle(shallow1+'knocktf_aucs.pkl')+pd.read_pickle(shallow2+'knocktf_aucs.pkl')

print(deep_pert)
print(deep_knocktf)
print(shallow_pert)
print(shallow_knocktf)

random_const = make_random_ranks(deep1)
deep_const = calculate_consistency(deep1)[1]+calculate_consistency(deep2)[1]
shallow_const = calculate_consistency(shallow1)[1]+calculate_consistency(shallow2)[1]

dorothea_pert = 0.639
dorothea_knocktf=0.60
title_sub_space = 1.02
swarmplot_color = 'lightsteelblue'
d_line_color = 'darkgrey'

PROPS = {
    'boxprops':{'facecolor':'white', 'edgecolor':'gray'},
    'medianprops':{'color':'gray'},
    'whiskerprops':{'color':'gray'},
    'capprops':{'color':'gray'}
}

#print(deep_pert)
#print(deep_knocktf)

"""
fig,ax = plt.subplots(1,3)
fig.set_figwidth(14)
fig.set_figheight(6)
plt.subplots_adjust(left=0.1,bottom=0.2,right=0.9,top=0.8,wspace=0.35,hspace=0)
"""

"""
def final_auc(fig):
    fig.subplots_adjust(left=0.1,bottom=0.2,right=0.9,top=0.8,wspace=0.35,hspace=0)
    ax = fig.subplots(1,2)
    pert_test(ax[0])
    ko_test(ax[1])
"""

def pert_test(a): 
    x_ticks = ["S-S","FC-G","VIPER"]
    with sns.color_palette("Paired"):
        sns.boxplot(data=[shallow_pert,deep_pert],ax=a,showfliers=False,**PROPS)
        sns.swarmplot(data=[shallow_pert,deep_pert,[dorothea_pert]],ax=a,color=swarmplot_color, edgecolor='k',linewidth=1)
    a.set_xticklabels(x_ticks)
    a.set_title("Perturbation Validation", fontsize='x-large',y=title_sub_space)
    a.axhline(y=dorothea_pert, color=d_line_color, linestyle='--',zorder=0)
    #a.axhline(y=0.5, color='lightcoral', linestyle='--',zorder=0)
    a.set_xlabel("Method")
    a.set_ylabel("ROC AUC")
    a.set_ylim(0.5,0.8)

def ko_test(a):
    x_ticks = ["S-S","FC-G","VIPER"]
    with sns.color_palette("Paired"):
        sns.boxplot(data=[shallow_knocktf,deep_knocktf],ax=a,showfliers=False,**PROPS)
        sns.swarmplot(data=[shallow_knocktf,deep_knocktf,[dorothea_knocktf]],ax=a, color=swarmplot_color,edgecolor='k',linewidth=1)
    a.set_xticklabels(x_ticks)
    a.set_title("Knock-out Validation", fontsize='x-large',y=title_sub_space)
    a.axhline(y=dorothea_knocktf, color=d_line_color, linestyle='--',zorder=0)
    #a.axhline(y=0.5, color='lightcoral', linestyle='--',zorder=0)
    a.set_xlabel("Method")
    a.set_ylabel("ROC AUC")
    a.set_ylim(0.4,0.8)

def final_const(a):
    #fig.subplots_adjust(left=0.1,bottom=0.2,right=0.9,top=0.8,wspace=0.35,hspace=0)
    #a = fig.subplots()
    x_ticks = ["S-S","FC-G","Random"]
    print('random',random_const)
    print('deep',deep_const)
    with sns.color_palette("Paired"):
        sns.boxplot(data=[shallow_const,deep_const,random_const],ax=a,showfliers=True,**PROPS)
        #sns.swarmplot(data=[shallow_const,deep_const,random_const],ax=a, edgecolor='k',linewidth=1)
    a.set_xticklabels(x_ticks)
    a.set_title("Consistency", fontsize='x-large',y=title_sub_space)
    a.set_xlabel("Method")
    a.set_ylabel("Kendall's W")
    a.set_ylim(-1,1)

"""
#CORRELATION BETWEEN EVAL AND FINAL EVAL 
a = ax[2]
a.scatter(deep_pert,deep_knocktf)
a.scatter(shallow_pert,shallow_knocktf)
corr = scipy.stats.pearsonr(deep_pert+shallow_pert,deep_knocktf+shallow_knocktf)[0]
print(corr)
a.set_xlabel("Perturbation Test AUC")
a.set_ylabel("KnockTF AUC")
a.set_ylim(0.5,0.8)
a.set_xlim(0.5,0.8)
a.set_title("pert vs. knocktf auc, corr: "+str(round(corr,2)))
"""

#fig.savefig("finaleval.png",dpi=300)

pvalues = []
labels = []
print("PVALUES")
stat, pval = ttest_ind(deep_knocktf,shallow_knocktf)
pvalues.append(pval)
labels.append('ko: deep,shallow,')
print('ko: deep,shallow,',str(pval))

stat,pval = ttest_ind(deep_pert,shallow_pert)
pvalues.append(pval)
labels.append('pert: deep,shallow,')
print('pert: deep,shallow,',str(pval))

stat, pval = ttest_1samp(deep_pert,dorothea_pert)
pvalues.append(pval)
labels.append('pert: deep,dor,')
print('pert: deep,dor,',str(pval))

stat, pval = ttest_1samp(shallow_pert,dorothea_pert)
pvalues.append(pval)
labels.append('pert: shallow,dor,')
print('pert: shallow,dor,',str(pval))

stat, pval = ttest_1samp(deep_knocktf,dorothea_knocktf)
pvalues.append(pval)
labels.append('ko: deep,dor,')
print('ko: deep,dor,',str(pval))

stat, pval = ttest_1samp(shallow_knocktf,dorothea_knocktf)
pvalues.append(pval)
labels.append('ko: shallow,dor,')
print('ko: shallow,dor,',str(pval))

p_adjusted = multipletests(pvalues,alpha=0.05,method='bonferroni')
print(p_adjusted,labels)

