import numpy as np
from scipy.stats import ttest_1samp, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#KO ROC AUC DATA

tffc_fc = [0.5437530384054449, 0.49491243739471574, 0.6001684548507082, 0.6159342460797504, 0.5950356694667104, 0.5253925902477078, 0.5919266034301479, 0.5260030977603419, 0.5576760014019061, 0.5959570835830007, 0.5317916134356876, 0.5959062079569478] 

tffc_shallow = [0.563441905687895, 0.660591740059468, 0.6232264191473245, 0.6283705102260008, 0.5958044567048422, 0.6046624684853761, 0.5717289798871692, 0.6146623554284293, 0.5520627239940757, 0.5972063628449652, 0.6585454093226758, 0.6122316310725713]

tffc_genefc = [0.6190037421849386, 0.6621575787724276, 0.5801630281172626, 0.614679313970447, 0.6017682106477032, 0.6369797967236097, 0.6527116708686165, 0.6432657629648053, 0.6935647985890493, 0.6716882793863268, 0.6430396490712371, 0.6239330250647251]


shallow_fc = [0.6006772111112368, 0.5699822500593549, 0.5050762569106059, 0.6312534623689953, 0.5516896360696882, 0.5973872539598194, 0.5784106454421092, 0.514934822670179, 0.5746345434195205, 0.49950254943414996, 0.5659856869905371, 0.5602989225672972]

shallow_shallow = [0.596624119569027, 0.6202982442256164, 0.6364540819210638, 0.6314456591785281, 0.594939571061944, 0.6165164893556886, 0.5883200868277353, 0.5485918757278041, 0.6340516218019016, 0.653497416648766, 0.6043967846604336, 0.6285231371041594]

shallow_genefc = [0.6674373381872448, 0.6367028072039886, 0.6567647624108264, 0.6703542074142744, 0.6175170433347278, 0.6252784027314556, 0.6462391606652271, 0.6299702660229958, 0.5909543136878046, 0.673570677550282, 0.593729861731354, 0.6503431278334899]


fc_fc= [0.438960554431267, 0.5820680376705749, 0.5137872946603205, 0.4792031746390657, 0.5496715695695922, 0.5941594781291337, 0.5760477552543216, 0.526364879990051, 0.5143130094628664, 0.5843291766062565, 0.5152005064951216, 0.5123740828255191]

fc_shallow = [0.6057138980904682, 0.6296254423353043, 0.6199081977592114, 0.6147528009858567, 0.645905642672214, 0.6396253292783576, 0.6499191642830494, 0.6450181456399587, 0.6437462549886378, 0.6071779855513222, 0.5751659110694056, 0.6411798622966389]

fc_genefc = [0.6659280279476772, 0.5719211766967021, 0.6272625521475167, 0.6431244417813252, 0.6430396490712371, 0.6206939435393608, 0.6174096392352828, 0.6967925744197352, 0.6260358842749094, 0.6429096335824355, 0.6267255316502922, 0.6424574057952991]


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
