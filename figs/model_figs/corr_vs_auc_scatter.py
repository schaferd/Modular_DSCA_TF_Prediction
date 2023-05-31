import matplotlib.pyplot as plt
from statsmodels.sandbox.stats.multicomp import multipletests
import copy
import scipy.stats
import math
import seaborn as sns
import pandas as pd
import numpy as np
from corr_data import corr_dict
from auc_data import auc_dict


#PLOT
#AUC vs CORR
plt.clf()
fig,ax = plt.subplots(3,3,sharex=True,sharey=True)
fig.set_figwidth(8)
fig.set_figheight(8)
fig.suptitle('Model Type Test Correlation vs. KO ROC AUC',fontsize='x-large')
fig.supylabel('KO ROC AUC',fontsize='x-large')
fig.supxlabel('Input vs. Ouput Test Correlation',fontsize='x-large')
plt.subplots_adjust(left=0.1,bottom=0.08,right=0.9,top=0.9,wspace=0.1,hspace=0.25)

#markers = ["." , "," , "o" , "v" , "^" , "<", ">",'s']
pcorr_dict = {}
s_enc_corr_list = []
s_enc_auc_list = []
full_corr_list = []
full_auc_list = []
pvalues = []
labels = []
keys = list(corr_dict.keys())
keys.sort()
keys.reverse()

title_space=01.0

for j in range(len(ax)):
        for i,a in enumerate(ax[j]):
            #for i,col in enumerate(corr_dict.keys()):
            col = keys[i+(j*len(ax[j]))]
            k = i+(j*len(ax[j]))
            min_len = min(len(corr_dict[col]),len(auc_dict[col]))
            pcorr = scipy.stats.pearsonr(corr_dict[col][:min_len],auc_dict[col][:min_len])
            print(col,str(pcorr))
            #a.scatter(corr_dict[col][:min_len],auc_dict[col][:min_len],marker=markers[k%len(markers)],label=col,edgecolors='black')
            a.scatter(corr_dict[col][:min_len],auc_dict[col][:min_len],label=col,edgecolors='black')
            pcorr_dict[col] = pcorr[0]
            labels.append(col)
            pvalues.append(pcorr[1])
            full_corr_list.extend(corr_dict[col][:min_len])
            full_auc_list.extend(auc_dict[col][:min_len]) 
            if col != 'fc_fc' :
                if col != 'tf_fc' :
                    if col != 'shallow_fc':
                        s_enc_corr_list.extend(corr_dict[col][:min_len])
                        s_enc_auc_list.extend(auc_dict[col][:min_len]) 
            #a.set_ylabel('KO ROC AUC')
            #a.set_xlabel('Input vs. Ouput Test Correlation')
            a.set_title(col+' corr: '+str(round(pcorr[0],2)), y=title_space)
            a.axhline(y=0.5, color='r', linestyle='--',alpha=0.4)
            a.set_ylim([0.4,0.8])
fig.savefig('auc_vs_corr.png',dpi=300)

p_adjusted = multipletests(pvalues,alpha=0.05,method='bonferroni')[1]
for i,pval in enumerate(p_adjusted):
    print(pval,labels[i])

print("pcorr")
print(pcorr_dict)
full_corr = scipy.stats.pearsonr(full_corr_list,full_auc_list)
s_enc_corr = scipy.stats.pearsonr(s_enc_corr_list,s_enc_auc_list)
print("full corr")
print(full_corr)
print(s_enc_corr)


plt.clf()
fig,ax = plt.subplots()
plt.subplots_adjust(left=0.13,bottom=0.13,right=0.9,top=0.83,wspace=0.1,hspace=0.3)
ax.scatter(s_enc_corr_list, s_enc_auc_list)
fig.suptitle('Model Type Test Correlation vs. KO ROC AUC,\nOnly Models with Sparse Encoders\n corr: '+ str(round(s_enc_corr[0],2)),fontsize='x-large')
fig.supylabel('KO ROC AUC',fontsize='x-large')
fig.supxlabel('Input vs. Ouput Test Correlation',fontsize='x-large')
fig.savefig('sparse_enc_auc_corr.png')


plt.clf()
fig,ax = plt.subplots()
plt.subplots_adjust(left=0.13,bottom=0.13,right=0.9,top=0.85,wspace=0.1,hspace=0.3)
ax.scatter(full_corr_list, full_auc_list)
fig.suptitle('Model Type Test Correlation vs. KO ROC AUC,\n corr: '+ str(round(full_corr[0],2)),fontsize='x-large')
fig.supylabel('KO ROC AUC',fontsize='x-large')
fig.supxlabel('Input vs. Ouput Test Correlation',fontsize='x-large')
fig.savefig('total_auc_corr.png')




dec_corr_dict = {}
enc_corr_dict = {}
dec_auc_dict = {}
enc_auc_dict = {}
for i in corr_dict.keys():
    split = i.split('_')
    if split[0] not in enc_corr_dict:
        enc_corr_dict[split[0]] = copy.deepcopy(corr_dict[i])
        enc_auc_dict[split[0]] = copy.deepcopy(auc_dict[i])
    else:
        enc_corr_dict[split[0]].extend(copy.deepcopy(corr_dict[i]))
        enc_auc_dict[split[0]].extend(copy.deepcopy(auc_dict[i]))

    if split[1] not in dec_corr_dict:
        dec_corr_dict[split[1]] = copy.deepcopy(corr_dict[i])
        dec_auc_dict[split[1]] = copy.deepcopy(auc_dict[i])
    else:
        dec_corr_dict[split[1]].extend(copy.deepcopy(corr_dict[i]))
        dec_auc_dict[split[1]].extend(copy.deepcopy(auc_dict[i]))


"""
#ENCODER
plt.clf()
fig,ax = plt.subplots()
markers = [ "v" , "o" ,"." ]
colors = ["w","r","k"]
keys = ['tf','shallow','fc']
for i,col in enumerate(keys):
    min_len = min(len(enc_corr_dict[col]),len(enc_auc_dict[col]))
    plt.scatter(enc_corr_dict[col][:min_len],enc_auc_dict[col][:min_len],marker=markers[i%len(markers)],label=col,edgecolors='black')
plt.legend(loc='best')
plt.ylabel('KO ROC AUC')
plt.xlabel('Input vs. Ouput Test Correlation')
plt.title('Encoder Test Correlation vs. KO ROC AUC')
plt.savefig('enc_auc_vs_corr.png')

#DECODER
plt.clf()
fig,ax = plt.subplots()
markers = [ "v" , "o" ,"." ]
colors = ["w","r","k"]
keys = ['gene','shallow','fc']
for i,col in enumerate(keys):
    min_len = min(len(dec_corr_dict[col]),len(dec_auc_dict[col]))
    plt.scatter(dec_corr_dict[col][:min_len],dec_auc_dict[col][:min_len],color=colors[i],marker=markers[i%len(markers)],label=col,edgecolors='black')
plt.legend(loc='best')
plt.ylabel('KO ROC AUC')
plt.xlabel('Input vs. Ouput Test Correlation')
plt.title('Decoder Test Correlation vs. KO ROC AUC')
plt.savefig('dec_auc_vs_corr.png')
"""
