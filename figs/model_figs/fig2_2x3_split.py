from auc_data import df as auc_df
from auc_data import auc_dict
from corr_data import df as corr_df
from corr_data import corr_dict
import numpy as np
from scipy.stats import ttest_1samp, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams.update({'font.size': 15})

fig,ax = plt.subplots()
subfigs = fig.subfigures(2,1)
#plt.tight_layout()
plt.subplots_adjust(left=0.1,bottom=0.1,right=0.8,top=0.8,wspace=3,hspace=3)
fig.set_figwidth(11)
fig.set_figheight(11)

print(auc_df)
encoders = ['e_shallow','e_tf','e_fc']
decoders = ['d_shallow','d_gene','d_fc']
print(decoders)
print(encoders)

decoder_name_dict = {'d_fc':'FC','d_shallow':'S','d_gene':'G'}

axBottom = subfigs[1].subplots(1,3,sharey=True)
for i, a in enumerate(axBottom):
    decoder_df = auc_df[auc_df['decoder']==decoders[i]]
    with sns.color_palette("Paired"):
        sns.boxplot(x='encoder',y='AUC',data=decoder_df,ax=a)
    a.set_xticklabels(['S','T','FC'])
    a.set_ylim(0.45, 0.78)
    a.yaxis.get_major_ticks()[-1].set_visible(False)
    if i == 0:
        a.set_ylabel('ROC AUC')
    else:
        a.set_ylabel('')
        a.get_yaxis().set_visible(False)
    a.set_xlabel('Encoder Module')
    a.set_title(decoder_name_dict[decoders[i]] + ' Decoder')
    a.axhline(y=0.5, color='r', linestyle='--',alpha=0.4)
subfigs[1].suptitle("TF Knock Out Prediction",fontsize='x-large',y=0.8)

axTop = subfigs[0].subplots(1,3,sharey=True)
for i, a in enumerate(axTop):
    decoder_df = corr_df[corr_df['decoder']==decoders[i]]
    with sns.color_palette("Paired"):
        sns.boxplot(x='encoder',y='Corr',data=decoder_df,ax=a)
    a.set_xticklabels(['S','T','FC'])
    a.set_ylim(0, 1)
    if i == 0:
        a.set_ylabel('Correlation')
    else:
        a.set_ylabel('')
        a.get_yaxis().set_visible(False)
    a.set_xlabel('Encoder Module')
    a.set_title(decoder_name_dict[decoders[i]] + ' Decoder')

subfigs[0].suptitle("Reconstruction Correlation", fontsize='x-large',y=0.99)
fig.savefig('model_boxplots.png', bbox_inches='tight')
    

print()
print("AUC MODEL PVALUES")
keys = list(auc_dict.keys())
for i,key in enumerate(keys):
    for j in keys[i+1:]:
        if key != j:
            stat, pval = ttest_ind(auc_dict[key],auc_dict[j])
            if pval <0.05:
                print(key,j,'pval',pval)

print()
print("CORR MODEL PVALUES")
keys = list(corr_dict.keys())
for i,key in enumerate(keys):
    for j in keys[i+1:]:
        if key != j:
            stat, pval = ttest_ind(corr_dict[key],corr_dict[j])
            if pval <0.05:
                print(key,j,'pval',pval)

"""
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
"""
