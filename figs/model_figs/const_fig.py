import numpy as np
from const_data import df, random_distances
from scipy.stats import ttest_1samp, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams.update({'font.size': 15})

fig,ax = plt.subplots(1,4,sharey=True, gridspec_kw={'width_ratios': [3,3,3,1.5]})
fig.set_figwidth(12)
fig.set_figheight(5)
#plt.tight_layout()
plt.subplots_adjust(left=0.2,bottom=0.2,right=0.8,top=0.8,wspace=0.1,hspace=2)

encoders = ['e_shallow','e_tf','e_fc']
decoders = ['d_shallow','d_gene','d_fc']
print(decoders)
print(encoders)

decoder_name_dict = {'d_fc':'FC','d_shallow':'S','d_gene':'G'}

for i, a in enumerate(ax[:3]):
    decoder_df =df[df['decoder']==decoders[i]]
    with sns.color_palette("Paired"):
        sns.boxplot(x='encoder',y='distance',data=decoder_df,ax=a,  boxprops=dict(alpha=.3))
        sns.swarmplot(data=decoder_df, x="encoder", y='distance',ax=a, edgecolor='k',linewidth=1)
    a.set_xticklabels(['S','T','FC'])
    if i == 0:
        a.set_ylabel('Kendall\'s W')
    else:
        a.set_ylabel('')
        #a.get_yaxis().set_visible(False)
    a.set_xlabel('Encoder\n Module')
    a.set_ylim(-1,1)
    a.set_title(decoder_name_dict[decoders[i]] + ' Decoder')

a = ax[3]
with sns.color_palette("Paired"):
    sns.boxplot(data=random_distances, ax=a, boxprops=dict(alpha=0.3))
    sns.swarmplot(data=random_distances,ax=a,edgecolor='k',linewidth=1)
    a.set_title("Random")
    a.get_xaxis().set_visible(False)
    

fig.suptitle("Ranking Consistency Between Trained Models", fontsize='x-large',y=0.95)
fig.savefig('const_boxplots.png', bbox_inches='tight')
    

