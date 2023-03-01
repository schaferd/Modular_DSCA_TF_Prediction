import numpy as np
from scipy.stats import ttest_1samp, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#TEST CORRELATION BETWEEN INPUT AND OUTPUT GENE EXPRESSION

tffc_fc = [0.7573196111013849, 0.7566821327503698, 0.7551391448908087, 0.7589744874318111, 0.7504908807469168, 0.7667399864958432, 0.75456019175948, 0.757779616197789,0.7398458561927099, 0.7585575051306866, 0.7541771351167665, 0.7610956260542072]

tffc_shallow = [0.376, 0.3856583507788242, 0.38167507888770913, 0.39540320380094285, 0.36960961978803064,0.3919639211487756, 0.3880746081090073, 0.3989148465452641, 0.37560803302779183, 0.38961847018420726, 0.37524194297557284, 0.3844929080364988]

tffc_genefc = [0.5513159553172646, 0.5164671307279449, 0.5750876697735549, 0.534502183882283, 0.5287112741663079, 0.5245304126973134, 0.5288043978849957, 0.5985543340321365, 0.5813431546573526, 0.5912450360434832, 0.5820258967453575, 0.5824781108366694]


shallow_fc = [0.754497268791375, 0.7507612192697471, 0.7553940593300448, 0.7512727611625616, 0.755606711844785, 0.751283466786714, 0.7579896153038909, 0.7550672071133255, 0.7535671395106613, 0.7341822150349419, 0.7515304286133278, 0.7524641366920838]

shallow_shallow = [0.37634661593804075, 0.3715698367352808, 0.36357986237891776, 0.37440754677367, 0.37063083654706425, 0.37697057085390345, 0.37559644425128824, 0.3449364511687285, 0.3743643254853444, 0.37820958110195246, 0.3748867951976996, 0.3749037562503841]

shallow_genefc = [0.5636663949384628, 0.5746842924615349, 0.5812027152530903, 0.5630923039566934, 0.5533105992584119, 0.5739821451532359, 0.5625173740528935, 0.5667870808499614, 0.5731178854545461, 0.5800270986802931, 0.5629768470937279, 0.5704050104396412]


fc_fc= [0.7971360143899338, 0.768347229938208, 0.7790052891209344, 0.7840844662313965, 0.7643394156165264, 0.7796865102347554, 0.787666525249964, 0.7757012324989767, 0.7765223634684326, 0.7844352606389299, 0.7920063840834506, 0.7842845929500403 ]

fc_shallow = [0.37791850922571824, 0.3903063418424452, 0.3956936287754252, 0.40314583501622847, 0.3912883058319394, 0.39651627843009996, 0.39458527139072114, 0.38465523439516175, 0.3945340058111708, 0.3945753763313034, 0.39807042811906473, 0.4023409749408271]

fc_genefc = [0.5875478561949806, 0.5773753514836577, 0.5686726388655958, 0.5713531142109631, 0.5700587813807111, 0.5877816048114224, 0.5801726230619043, 0.5693530365231269, 0.5674313749859693, 0.5769667790341269, 0.5757354706578587, 0.5683816409988419]


corr_dict = {'tf_fc':tffc_fc,'tf_shallow':tffc_shallow,'tf_gene':tffc_genefc, 'shallow_fc':shallow_fc, 'shallow_shallow':shallow_shallow, 'shallow_gene':shallow_genefc,'fc_fc':fc_fc,'fc_shallow':fc_shallow,'fc_gene':fc_genefc}

decoder = ['d_fc','d_shallow','d_gene']

fc_df = pd.concat([pd.DataFrame({'d_fc':fc_fc}),pd.DataFrame({'d_shallow':fc_shallow}),pd.DataFrame({'d_gene':fc_genefc})],axis=1)

fc_df = fc_df.melt(value_vars=decoder,value_name='Corr',var_name='decoder').dropna()
fc_df['encoder'] = 'e_fc'

shallow_df = pd.concat([pd.DataFrame({'d_fc':shallow_fc}),pd.DataFrame({'d_shallow':shallow_shallow}),pd.DataFrame({'d_gene':shallow_genefc})],axis=1)
shallow_df = shallow_df.melt(value_vars=decoder,value_name='Corr',var_name='decoder').dropna()
shallow_df['encoder'] = 'e_shallow'

tffc_df = pd.concat([pd.DataFrame({'d_fc':tffc_fc}),pd.DataFrame({'d_shallow':tffc_shallow}),pd.DataFrame({'d_gene':tffc_genefc})],axis=1)
tffc_df = tffc_df.melt(value_vars=decoder,value_name='Corr',var_name='decoder').dropna()
tffc_df['encoder'] = 'e_tf'


df = pd.concat([fc_df,shallow_df,tffc_df],axis=0)

fig,ax = plt.subplots()
sns.boxplot(x='encoder',y='Corr',data=df,hue='decoder',ax=ax)
ax.set_ylim(0.3, 0.8)
plt.ylabel('Test Input vs. Ouput Correlation')
plt.savefig('model_corr_boxplot.png')

print()
print("MODEL PVALUES")
keys = list(corr_dict.keys())
for i,key in enumerate(keys):
    for j in keys[i+1:]:
        if key != j:
            stat, pval = ttest_ind(corr_dict[key],corr_dict[j])
            if pval <0.05:
                print(key,j,'pval',pval)

plt.clf()
enc_df = df.drop(columns=['decoder'])
fig,ax = plt.subplots()
sns.boxplot(x='encoder',y='Corr',data=enc_df,ax=ax)
ax.set_ylim(0.3, 0.8)
plt.ylabel('Test Input vs. Ouput Correlation')
plt.xlabel('encoder type')
plt.savefig('enc_model_corr_boxplot.png')

print()
print("ENCODER PVALUES")
enc_keys = list(set(list(enc_df['encoder'])))
for i,key in enumerate(enc_keys):
    for j in enc_keys[i+1:]:
        if key != j:
            stat, pval = ttest_ind(enc_df[enc_df['encoder'] == key]["Corr"],enc_df[enc_df['encoder'] == j]["Corr"])
            if pval <0.05:
                print(key,j,'pval',pval)

plt.clf()
dec_df = df.drop(columns=['encoder'])
fig,ax = plt.subplots()
sns.boxplot(x='decoder',y='Corr',data=dec_df,ax=ax)
ax.set_ylim(0.3, 0.8)
plt.ylabel('Test Input vs. Ouput Correlation')
plt.xlabel('decoder type')
plt.savefig('dec_model_corr_boxplot.png')

print()
print("DECODER PVALUES")
dec_keys = list(set(list(dec_df['decoder'])))
for i,key in enumerate(dec_keys):
    for j in dec_keys[i+1:]:
        if key != j:
            stat, pval = ttest_ind(dec_df[dec_df['decoder'] == key]["Corr"],dec_df[dec_df['decoder'] == j]["Corr"])
            if pval <0.05:
                print(key,j,'pval',pval)
