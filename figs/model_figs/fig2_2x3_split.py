from auc_data import df as auc_df
from statsmodels.sandbox.stats.multicomp import multipletests
from auc_data import auc_dict
from corr_data import df as corr_df
from corr_data import corr_dict
import numpy as np
from scipy.stats import ttest_1samp, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#plt.rcParams.update({'font.size': 15})

#fig,ax = plt.subplots()
#fig.set_figwidth(11)
#fig.set_figheight(8)

print(auc_df)
encoders = ['e_shallow','e_tf','e_fc']
decoders = ['d_shallow','d_gene','d_fc']
print(decoders)
print(encoders)

decoder_name_dict = {'d_fc':'FC','d_shallow':'S','d_gene':'G'}

title_sub_space=0.97

swarmplot_color = 'white'
d_line_color = 'darkgrey'

PROPS = {
    #    'boxprops':{'alpha':0.6},
    'boxprops':{'facecolor':'white', 'edgecolor':'gray'},
    'medianprops':{'color':'gray'},
    'whiskerprops':{'color':'gray'},
    'capprops':{'color':'gray'}
}

def make_model_boxplots(fig,axlabel_font_size,title_font_size,subtitle_font_size):

    subfigs = fig.subfigures(2,1)
    #plt.tight_layout()
    plt.subplots_adjust(left=0.2,bottom=0.2,right=0.8,top=0.8,wspace=0.09,hspace=2)

    axBottom = subfigs[1].subplots(1,3,sharey=True)
    for i, a in enumerate(axBottom):
        decoder_df = auc_df[auc_df['decoder']==decoders[i]]
        with sns.color_palette("Paired"):
            sns.boxplot(x='encoder',y='AUC',data=decoder_df,ax=a,**PROPS)
            sns.swarmplot(data=decoder_df, x="encoder", y='AUC',ax=a,edgecolor='gray',linewidth=1,alpha=0.8)
        a.set_xticklabels(['S','T','FC'])
        a.set_ylim(0.40, 0.78)
        a.yaxis.get_major_ticks()[-1].set_visible(False)
        if i == 0:
            a.set_ylabel('ROC AUC')
        else:
            a.set_ylabel('')
            a.get_yaxis().set_visible(False)
        a.set_xlabel('Encoder Module')
        a.set_title(decoder_name_dict[decoders[i]] + ' Decoder')
        a.axhline(y=0.5, color=d_line_color, linestyle='--')
    subfigs[1].suptitle("TF Perturbation Prediction",fontsize='x-large',y=title_sub_space)

    axTop = subfigs[0].subplots(1,3,sharey=True)
    for i, a in enumerate(axTop):
        decoder_df = corr_df[corr_df['decoder']==decoders[i]]
        with sns.color_palette("Paired"):
            #sns.boxplot(x='encoder',y='Corr',data=decoder_df,ax=a,**PROPS)
            sns.swarmplot(data=decoder_df, x="encoder", y='Corr',ax=a, edgecolor='gray',linewidth=1,alpha=0.8)
        a.set_xticklabels(['S','T','FC'])
        a.set_ylim(0, 1)
        if i == 0:
            a.set_ylabel('Correlation')
        else:
            a.set_ylabel('')
            a.get_yaxis().set_visible(False)
        a.set_xlabel('Encoder Module')
        a.set_title(decoder_name_dict[decoders[i]] + ' Decoder')

    subfigs[0].suptitle("Reconstruction Correlation", fontsize='x-large',y=title_sub_space)
    #fig.savefig('model_boxplots.png', bbox_inches='tight',dpi=300)


df_auc = auc_df.sort_values(by=['encoder','decoder'],ascending=False)
df_corr = corr_df.sort_values(by=['encoder','decoder'],ascending=False)

df = pd.concat([df_corr,df_auc.drop(columns=['encoder','decoder'])],axis=1)
print(df)
    
"""
plt.clf()

fig, ax = plt.subplots()
ax.scatter(df['Corr'],df['AUC'])
ax.set_ylabel("ROC AUC")
ax.set_xlabel("Correlation")
fig.savefig("rocvscorr.png",dpi=300)

"""

print()
pvals = []
labels = []
keys = list(auc_dict.keys())
for i,key in enumerate(keys):
    for j in keys[i+1:]:
        if key != j:
            stat, pval = ttest_ind(auc_dict[key],auc_dict[j])
            pvals.append(pval)
            labels.append(key+', '+j)

print("AUC MODEL PVALUES")
p_adjusted = multipletests(pvals,alpha=0.05,method='bonferroni')[1]
for i in range(len(p_adjusted)):
    if p_adjusted[i] <0.05:
        print(labels[i],p_adjusted[i])

pvals = []
labels = []
keys = list(corr_dict.keys())

diff_decoder = []
same_decoder = []

for i,key in enumerate(keys):
    for j in keys[i+1:]:
        if key != j:
            stat, pval = ttest_ind(corr_dict[key],corr_dict[j])
            pvals.append(pval)
            labels.append(key+', '+j)
            dec1 = key.split('_')[1]
            dec2 = j.split('_')[1]
            if dec1 == dec2:
                same_decoder.append(pval)
            else:
                diff_decoder.append(pval)



print("\nCORR MODEL PVALUES")
p_adjusted = multipletests(pvals,alpha=0.05,method='bonferroni')[1]
for i in range(len(p_adjusted)):
    if p_adjusted[i] <0.05:
        print(labels[i],p_adjusted[i])

p_adjusted = multipletests(diff_decoder+same_decoder,alpha=0.05,method='bonferroni')[1]
same_decoder = p_adjusted[len(diff_decoder):]
diff_decoder = p_adjusted[:len(diff_decoder)]

print("diff decoder",np.mean(np.array(diff_decoder)))
print("same decoder",np.mean(np.array(same_decoder)))
