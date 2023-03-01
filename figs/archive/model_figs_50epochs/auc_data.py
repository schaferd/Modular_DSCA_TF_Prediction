import numpy as np
from scipy.stats import ttest_1samp, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#KO ROC AUC DATA

tffc_fc = [0.4771003154288816, 0.4658568020712033, 0.5341884207075104, 0.5565228205447084, 0.508643203581644, 0.5540073034787624, 0.5127528235972459, 0.5538829408372997, 0.530259691806763, 0.5712032650846232] 

tffc_shallow = [0.5875478561949806, 0.5773753514836577, 0.5686726388655958, 0.5713531142109631, 0.5700587813807111, 0.5877816048114224, 0.5801726230619043, 0.5693530365231269, 0.5674313749859693, 0.5769667790341269, 0.5757354706578587, 0.5683816409988419]

tffc_genefc = [0.6323896846841754, 0.6356909475302709, 0.6169461057534681, 0.6310895297961583, 0.6287492509977275, 0.6414511989689206, 0.6645317746548938, 0.6353743880792754, 0.6399249301873354, 0.613300019219681]


shallow_fc = [0.5532385162406306, 0.5084905767034855, 0.6152954743304203, 0.542017614272309, 0.5549513289844095, 0.5554883494816337, 0.6124803563554962, 0.5822941515641429, 0.5814123073792268, 0.5724016687205346] 

shallow_shallow = [0.6089982023945462, 0.5969011090886479, 0.691789804524539, 0.5895976303263954, 0.6026952776113328, 0.6904161626211122, 0.6752043504313123, 0.6503261692914721, 0.5947360685577324, 0.623073792269166]

shallow_genefc = [0.658522797933319, 0.6146680082757684, 0.6566460526167031, 0.6332150003956993, 0.6202530214469029, 0.628127437790415, 0.6226667872607433, 0.6846276469457666, 0.6023900238550157, 0.6385512882839086] 


fc_fc= [0.5464268351968887, 0.5297283241568778, 0.5034086669455404, 0.49364054674339464, 0.5510112943889838, 0.5202824162530666, 0.47456783982091777, 0.6095239171970921, 0.4736690370939842, 0.5510960870990718]

fc_shallow = [0.6075510734757097, 0.5602932697199579, 0.6611457190987099, 0.680167550395134, 0.605793037953217, 0.6423273903064973, 0.5746741133508948, 0.6111236729940871, 0.6697493527489796, 0.6086929486382291]

fc_genefc = [0.6196029440028943, 0.5965788967903134, 0.6580083888254513, 0.6608461181897322, 0.6103153158245808, 0.6543905665283604, 0.6324405603102283, 0.6164316966456004, 0.6311234468801935, 0.6980531593763779]



auc_dict = {'tf_fc':tffc_fc,'tf_shallow':tffc_shallow,'tf_gene':tffc_genefc, 'shallow_fc':shallow_fc, 'shallow_shallow':shallow_shallow, 'shallow_gene':shallow_genefc,'fc_fc':fc_fc,'fc_shallow':fc_shallow,'fc_gene':fc_genefc}

decoder = ['d_fc','d_shallow','d_gene']

fc_df = pd.concat([pd.DataFrame({'d_fc':fc_fc}),pd.DataFrame({'d_shallow':fc_shallow}),pd.DataFrame({'d_gene':fc_genefc})],axis=1)

fc_df = fc_df.melt(value_vars=decoder,value_name='AUC',var_name='decoder').dropna()
fc_df['encoder'] = 'e_fc'

shallow_df = pd.concat([pd.DataFrame({'d_fc':shallow_fc}),pd.DataFrame({'d_shallow':shallow_shallow}),pd.DataFrame({'d_gene':shallow_genefc})],axis=1)
shallow_df = shallow_df.melt(value_vars=decoder,value_name='AUC',var_name='decoder').dropna()
shallow_df['encoder'] = 'e_shallow'

tffc_df = pd.concat([pd.DataFrame({'d_fc':tffc_fc}),pd.DataFrame({'d_shallow':tffc_shallow}),pd.DataFrame({'d_gene':tffc_genefc})],axis=1)
tffc_df = tffc_df.melt(value_vars=decoder,value_name='AUC',var_name='decoder').dropna()
tffc_df['encoder'] = 'e_tf'


df = pd.concat([fc_df,shallow_df,tffc_df],axis=0)

fig,ax = plt.subplots()
sns.boxplot(x='encoder',y='AUC',data=df,hue='decoder',ax=ax)
ax.set_ylim(0.3, 0.8)
plt.ylabel('KO ROC AUC')
plt.savefig('model_auc_boxplot.png')

print()
print("MODEL PVALUES")
keys = list(auc_dict.keys())
for i,key in enumerate(keys):
    for j in keys[i+1:]:
        if key != j:
            stat, pval = ttest_ind(auc_dict[key],auc_dict[j])
            if pval <0.05:
                print(key,j,'pval',pval)

plt.clf()
enc_df = df.drop(columns=['decoder'])
fig,ax = plt.subplots()
sns.boxplot(x='encoder',y='AUC',data=enc_df,ax=ax)
ax.set_ylim(0.3, 0.8)
plt.ylabel('KO ROC AUC')
plt.xlabel('encoder type')
plt.savefig('enc_model_auc_boxplot.png')

print()
print("ENCODER PVALUES")
enc_keys = list(set(list(enc_df['encoder'])))
for i,key in enumerate(enc_keys):
    for j in enc_keys[i+1:]:
        if key != j:
            stat, pval = ttest_ind(enc_df[enc_df['encoder'] == key]["AUC"],enc_df[enc_df['encoder'] == j]["AUC"])
            if pval <0.05:
                print(key,j,'pval',pval)

plt.clf()
dec_df = df.drop(columns=['encoder'])
fig,ax = plt.subplots()
sns.boxplot(x='decoder',y='AUC',data=dec_df,ax=ax)
ax.set_ylim(0.3, 0.8)
plt.ylabel('KO ROC AUC')
plt.xlabel('decoder type')
plt.savefig('dec_model_auc_boxplot.png')

print()
print("DECODER PVALUES")
dec_keys = list(set(list(dec_df['decoder'])))
for i,key in enumerate(dec_keys):
    for j in dec_keys[i+1:]:
        if key != j:
            stat, pval = ttest_ind(dec_df[dec_df['decoder'] == key]["AUC"],dec_df[dec_df['decoder'] == j]["AUC"])
            if pval <0.05:
                print(key,j,'pval',pval)
