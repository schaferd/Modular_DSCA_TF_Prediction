from auc_data import df as auc_df
from auc_data import auc_dict
from corr_data import df as corr_df
from corr_data import corr_dict
import numpy as np
from scipy.stats import ttest_1samp, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


fig,ax = plt.subplots(2,3)
plt.subplots_adjust(wspace=0.15, hspace=0.3)
fig.set_figwidth(15)
fig.set_figheight(10)

sns.boxplot(x='encoder',y='AUC',data=auc_df,hue='decoder',ax=ax[0,0])
label_list = [t for t in ax[0,0].get_legend_handles_labels()]
label_list[1] = ["Shallow", "G", "FC"]
ax[0,0].legend(loc='lower left',title="Decoder Module",handles= label_list[0],labels = label_list[1])
ax[0,0].axhline(y=0.5, color='r', linestyle='--',alpha=0.4)
ax[0,0].set_ylim(0.3, 0.8)
ax[0,0].set_ylabel('KO ROC AUC')
ax[0,0].set_xlabel('Encoder Module')
ax[0,0].set_xticklabels(['Shallow','T','FC'])
ax[0,0].set_title('Module Combination vs. KO ROC AUC')
#ax1.savefig('model_auc_boxplot.png')


print()
print("AUC MODEL PVALUES")
keys = list(auc_dict.keys())
for i,key in enumerate(keys):
    for j in keys[i+1:]:
        if key != j:
            stat, pval = ttest_ind(auc_dict[key],auc_dict[j])
            if pval <0.05:
                print(key,j,'pval',pval)

enc_df = auc_df.drop(columns=['decoder'])
sns.boxplot(x='encoder',y='AUC',data=enc_df,ax=ax[0,1])
ax[0,1].axhline(y=0.5, color='r', linestyle='--',alpha=0.4)
ax[0,1].set_ylim(0.3, 0.8)
ax[0,1].set_ylabel('')
#ax[0,1].get_yaxis().set_visible(False)
ax[0,1].set_xlabel('Encoder Module')
ax[0,1].set_title('Encoder Module vs. KO ROC AUC')
ax[0,1].set_xticklabels(['Shallow','T','FC'])
#plt.savefig('enc_model_auc_boxplot.png')

print()
print("AUC ENCODER PVALUES")
enc_keys = list(set(list(enc_df['encoder'])))
for i,key in enumerate(enc_keys):
    for j in enc_keys[i+1:]:
        if key != j:
            stat, pval = ttest_ind(enc_df[enc_df['encoder'] == key]["AUC"],enc_df[enc_df['encoder'] == j]["AUC"])
            if pval <0.05:
                print(key,j,'pval',pval)

dec_df = auc_df.drop(columns=['encoder'])
sns.boxplot(x='decoder',y='AUC',data=dec_df,ax=ax[0,2])
ax[0,2].axhline(y=0.5, color='r', linestyle='--',alpha=0.4)
ax[0,2].set_ylim(0.3, 0.8)
ax[0,2].set_ylabel('')
#ax[0,2].get_yaxis().set_visible(False)
ax[0,2].set_xlabel('Decoder Module')
ax[0,2].set_title('Decoder Module vs. KO ROC AUC')
ax[0,2].set_xticklabels(['Shallow','G','FC'])
#fig.savefig('model_auc_boxplot.png')

print()
print("AUC DECODER PVALUES")
dec_keys = list(set(list(dec_df['decoder'])))
for i,key in enumerate(dec_keys):
    for j in dec_keys[i+1:]:
        if key != j:
            stat, pval = ttest_ind(dec_df[dec_df['decoder'] == key]["AUC"],dec_df[dec_df['decoder'] == j]["AUC"])
            if pval <0.05:
                print(key,j,'pval',pval)



#############################################################
####CORR PLOTS
############################################################

sns.boxplot(x='encoder',y='Corr',data=corr_df,hue='decoder',ax=ax[1,0])
label_list = [t for t in ax[1,0].get_legend_handles_labels()]
label_list[1] = ["Shallow", "G", "FC"]
ax[1,0].legend(loc='lower right',title="Decoder Module",handles= label_list[0],labels = label_list[1])
ax[1,0].set_ylim(0.3,1)
ax[1,0].set_ylabel('Test Input vs. Ouput Correlation')
ax[1,0].set_xlabel('Encoder Module')
ax[1,0].set_title('Module Combination vs. Test Correlation')
ax[1,0].set_xticklabels(['Shallow','T','FC'])

print()
print("CORR MODEL PVALUES")
keys = list(corr_dict.keys())
for i,key in enumerate(keys):
    for j in keys[i+1:]:
        if key != j:
            stat, pval = ttest_ind(corr_dict[key],corr_dict[j])
            if pval <0.05:
                print(key,j,'pval',pval)

enc_df = corr_df.drop(columns=['decoder'])
sns.boxplot(x='encoder',y='Corr',data=enc_df,ax=ax[1,1])
ax[1,1].set_ylim(0.3, 1)
ax[1,1].set_xlabel('Encoder Module')
ax[1,1].set_ylabel('')
ax[1,1].set_title('Encoder Module vs. Test Correlation')
ax[1,1].set_xticklabels(['Shallow','T','FC'])

print()
print("CORR ENCODER PVALUES")
enc_keys = list(set(list(enc_df['encoder'])))
for i,key in enumerate(enc_keys):
    for j in enc_keys[i+1:]:
        if key != j:
            stat, pval = ttest_ind(enc_df[enc_df['encoder'] == key]["Corr"],enc_df[enc_df['encoder'] == j]["Corr"])
            if pval <0.05:
                print(key,j,'pval',pval)

dec_df = corr_df.drop(columns=['encoder'])
sns.boxplot(x='decoder',y='Corr',data=dec_df,ax=ax[1,2])
ax[1,2].set_ylim(0.3, 1)
ax[1,2].set_xlabel('Decoder Module')
ax[1,2].set_ylabel('')
ax[1,2].set_title('Decoder Module vs. Test Correlation')
ax[1,2].set_xticklabels(['Shallow','G','FC'])
fig.savefig('model_boxplots.png')

print()
print("CORR DECODER PVALUES")
dec_keys = list(set(list(dec_df['decoder'])))
for i,key in enumerate(dec_keys):
    for j in dec_keys[i+1:]:
        if key != j:
            stat, pval = ttest_ind(dec_df[dec_df['decoder'] == key]["Corr"],dec_df[dec_df['decoder'] == j]["Corr"])
            if pval <0.05:
                print(key,j,'pval',pval)
