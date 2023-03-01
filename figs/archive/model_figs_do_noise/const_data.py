import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams.update({'font.size': 12})
plt.tight_layout()

const_dict = {'fc_fc':0,'fc_gene':0.312,'fc_shallow':0.4431,'shallow_fc':0.2870,'shallow_gene':0.520,'shallow_shallow':0.48,'tf_gene':0.534, 'tf_fc':0.4354, 'tf_shallow':0.705}

fc_enc = {'fc':[0.07245],'gene':[0.312],'shallow':[0.4431],'encoder':'fc'}
tf_enc = {'gene':[0.534], 'fc':[0.4354], 'shallow':[0.705],'encoder':'tf'}
shallow_enc = {'fc':[0.2870],'gene':[0.520],'shallow':[0.48], 'encoder':'shallow'}

enc_order = ['shallow','tf','fc']
dec_order = ['shallow','gene','fc']



df = pd.concat([pd.DataFrame(shallow_enc),pd.DataFrame(tf_enc),pd.DataFrame(fc_enc)],axis=0)
print(df)
df = df.melt(value_vars=["shallow","gene","fc"], id_vars=['encoder'],value_name="Consistency", var_name="decoder")
df.decoder = df.decoder.astype("category")
df.decoder = df.decoder.cat.set_categories(dec_order)
df.encoder = df.encoder.astype("category")
df.encoder = df.encoder.cat.set_categories(enc_order)
df = df.pivot(index = 'encoder',columns = 'decoder').loc[::-1]
print(df)

plt.subplots_adjust(bottom=0.15)
fig, ax = plt.subplots()
fig.set_figwidth(4)
fig.set_figheight(3)
plt.subplots_adjust(wspace=0.08, hspace=0.2)
sns.heatmap(df,annot=True,cmap="crest",vmin=0,vmax=1).collections[0].colorbar.set_label("AUC")
ax.set_yticklabels(['FC','T','S'],rotation=0,fontsize=10)
ax.set_xticklabels(['S','G','FC'],rotation=0,fontsize=10)
ax.set_title("Pred. TF Consistency")
ax.set_ylabel('Encoder')
ax.set_xlabel('Decoder')
fig.savefig('consistency_heatmap.png', bbox_inches="tight")


rand_std = pd.read_pickle('/nobackup/users/schaferd/ae_project_outputs/model_eval/___fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_do0.15_noise0.15_rel_conn10_2-1_8.53.47/consistency_rand_std.pkl')

fc_genefc_std = pd.read_pickle('/nobackup/users/schaferd/ae_project_outputs/model_eval/___fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_do0.15_noise0.15_rel_conn10_2-1_8.53.47/consistency_std.pkl')
fc_shallow_std = pd.read_pickle('/nobackup/users/schaferd/ae_project_outputs/model_eval/___fc-shallow_epochs100_batchsize128_enlr0.0001_delr0.001_moa1.0_do0.15_noise0.15_rel_conn10_2-1_8.55.49/consistency_std.pkl')

shallow_fc_std = pd.read_pickle('/nobackup/users/schaferd/ae_project_outputs/model_eval/___shallow-fc_epochs100_batchsize128_enlr0.001_delr0.0001_moa1.0_do0.15_noise0.15_rel_conn10_2-1_8.52.45/consistency_std.pkl')
shallow_genefc_std = pd.read_pickle('/nobackup/users/schaferd/ae_project_outputs/model_eval/___shallow-genefc_epochs100_batchsize128_enlr0.01_delr0.01_moa1.0_do0.15_noise0.15_rel_conn10_1-31_20.0.4/consistency_std.pkl')
shallow_shallow_std = pd.read_pickle('/nobackup/users/schaferd/ae_project_outputs/model_eval/___shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_moa1.0_do0.15_noise0.15_rel_conn10_2-1_1.32.25/consistency_std.pkl')

tffc_fc_std = pd.read_pickle('/nobackup/users/schaferd/ae_project_outputs/model_eval/___tffc-fc_epochs100_batchsize128_enlr0.001_delr0.0001_moa1.0_do0.15_noise0.15_rel_conn10_1-31_19.59.5/consistency_std.pkl')
tffc_genefc_std = pd.read_pickle('/nobackup/users/schaferd/ae_project_outputs/model_eval/___tffc-genefc_epochs100_batchsize128_enlr0.001_delr0.01_moa1.0_do0.15_noise0.15_rel_conn10_1-31_19.57.33/consistency_std.pkl')
tffc_shallow_std = pd.read_pickle('/nobackup/users/schaferd/ae_project_outputs/model_eval/___tffc-shallow_epochs100_batchsize128_enlr0.001_delr0.01_moa1.0_do0.15_noise0.15_rel_conn10_1-31_19.58.39/consistency_std.pkl')

const_std_dict={'fc_gene':fc_genefc_std,'fc_shallow':fc_shallow_std,'shallow_fc':shallow_fc_std,'shallow_gene':shallow_genefc_std,'shallow_shallow':shallow_shallow_std,'tf_gene':tffc_genefc_std, 'tf_fc':tffc_fc_std, 'tf_shallow':tffc_shallow_std}
print(const_std_dict)

"""
keys = const_dict.keys()
auc_std = [np.std(auc_dict[col]) for col in keys]
const = [const_dict[col] for col in keys]
df = pd.DataFrame({"model":keys,"auc_std":auc_std,"const":const})
corr = df["const"].corr(df["auc_std"])
a, b = np.polyfit(df["const"],df["auc_std"], 1)
x = np.arange(min(df["const"]),max(df["const"]),0.005)

plt.clf()
fig,ax = plt.subplots()
plt.plot(x, a*x+b, color='0.8',linestyle='dashed',zorder=1)
markers = ["." , "," , "o" , "v" , "^" , "<", ">"]
for i,col in enumerate(auc_dict.keys()):
    if col in const_dict:
        plt.scatter(const_dict[col],np.std(auc_dict[col]),marker=markers[i%len(markers)],label=col,edgecolors='black',zorder=2)

plt.legend(loc='best')
plt.xlabel('Consistency')
plt.ylabel('KO ROC AUC Std')
plt.title('Model Type Consistency vs. KO ROC AUC Std, corr: '+str(round(corr,2)))
plt.savefig('const_vs_aucStd.png')
"""


