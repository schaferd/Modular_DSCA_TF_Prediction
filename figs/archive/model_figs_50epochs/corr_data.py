import numpy as np
from scipy.stats import ttest_1samp, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#TEST CORRELATION BETWEEN INPUT AND OUTPUT GENE EXPRESSION

tffc_fc = [0.8190033453488818, 0.8581848960893329, 0.8634089931567508, 0.8568745705322707, 0.857790723815595, 0.8651907372646958, 0.8508345760695321, 0.4329880224366441, 0.8574097646385521, 0.8617304544253482]

tffc_shallow =[0.4605244885351239, 0.46631596818343174, 0.4532093441289146, 0.47157572617116134, 0.45943810977520594, 0.457657697828943, 0.43757242946112485, 0.43965948809546507, 0.44894789071019187, 0.4588601404964464]
 
tffc_genefc = [0.5884623309984288, 0.5146394114241075, 0.5652776903891031, 0.5263965623752331, 0.5256807868552626, 0.5682250819473069, 0.5293190457471725, 0.5800199154113358, 0.5248026199056499, 0.5835090886303478]

shallow_fc = [0.855421745202516, 0.8505009165975744, 0.8546639575956463, 0.8520265893461532, 0.8547187585222322, 0.8448246274521889, 0.8534184183864484, 0.8588303071530277, 0.8543710023938266, 0.8634455459370006]

shallow_shallow =[0.5015138363740387, 0.5228434935947968, 0.5050148005818743, 0.5061047063544196, 0.5078799246743639, 0.5243784210749769, 0.5140981272111383, 0.4992472658933742, 0.5227407251047521, 0.5301052416324785]
 
shallow_genefc =[0.5661950441444354, 0.5675897417728798, 0.5525494106926971, 0.568382645374164, 0.5527346914709014, 0.5486268157910583, 0.5453532659104975, 0.5647545856848984, 0.5795747192246185, 0.5762121485293655]

fc_fc= [0.8774734940824335, 0.8822430667725883, 0.8781282659037863, 0.8747786838880991, 0.8696219937820643, 0.8720435264989421, 0.8780965519664438, 0.8771897608140651, 0.809560791038632, 0.8770281743222692]
 
fc_shallow = [0.5266211616385337, 0.5256549029357778, 0.5353853593910585, 0.5181554272835774, 0.5315657774920263, 0.5244846477482894, 0.532073595655024, 0.5276121046695099, 0.5119757831188955, 0.5317236851142761]

fc_genefc = [0.5883550643596404, 0.5812793588077492, 0.5686958846937097, 0.5916643107447418, 0.5701865835801457, 0.5836795288630713, 0.5791366054229836, 0.5787887671555445, 0.589451718506006, 0.5893639908430632]


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
ax.set_ylim(0.3,1)
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
ax.set_ylim(0.3, 1)
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
ax.set_ylim(0.3, 1)
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
