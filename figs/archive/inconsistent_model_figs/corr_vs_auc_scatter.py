import matplotlib.pyplot as plt
import copy
import math
import seaborn as sns
import pandas as pd
import numpy as np
from corr_data import corr_dict
from auc_data import auc_dict
from const_data import const_dict



keys = list(corr_dict.keys())
points = {}
for index,key1 in enumerate(keys):
    for i in range(len(corr_dict[key1])):
        point = (corr_dict[key1][i],auc_dict[key1][i])
        if point in points:
            points[point] += 1
        else:
            points[point] = 1
loc = np.array(list(points.keys())).T
num = [points[i] for i in points.keys()]
points = np.vstack([loc,num])
points = pd.DataFrame(points.T,columns=['corr','auc','freq'])
print(points)

plt.clf()
sns.lmplot(x="corr",y="auc",data=points,fit_reg=False,hue='freq',legend=False)
plt.legend(loc='best')
plt.ylabel("KO ROC AUC")
plt.xlabel('Input vs. Ouput Test Correlation')
plt.title('Test Correlation vs. KO ROC AUC (Repeated points)')
plt.savefig('repeated_auc_vs_corr.png', bbox_inches='tight',dpi=100)



#PLOT
#AUC vs CORR
plt.clf()
fig,ax = plt.subplots()
markers = ["." , "," , "o" , "v" , "^" , "<", ">",'s']
for i,col in enumerate(corr_dict.keys()):
    min_len = min(len(corr_dict[col]),len(auc_dict[col]))
    plt.scatter(corr_dict[col][:min_len],auc_dict[col][:min_len],marker=markers[i%len(markers)],label=col,edgecolors='black')
plt.legend(loc='best')
plt.ylabel('KO ROC AUC')
plt.xlabel('Input vs. Ouput Test Correlation')
plt.title('Model Type Test Correlation vs. KO ROC AUC')
ax.set_ylim([0,0.8])
plt.savefig('auc_vs_corr.png')



dec_corr_dict = {}
enc_corr_dict = {}
dec_auc_dict = {}
enc_auc_dict = {}
for i in corr_dict.keys():
    split = i.split('_')
    print(split)
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
