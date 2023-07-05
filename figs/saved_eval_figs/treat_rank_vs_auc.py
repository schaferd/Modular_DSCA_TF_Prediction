import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

cutoffs = [0.05,0.1,0.2,0.3,0.4,0.5]
s_s_avg_aucs = pd.read_pickle('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/treated_ranked/shallow_shallow/rank_vs_auc_avg_aucs.pkl')
s_s_auc_dict = pd.read_pickle('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/treated_ranked/shallow_shallow/rank_vs_auc_auc_dict.pkl')
fc_g_avg_aucs = pd.read_pickle('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/treated_ranked/fc_g/rank_vs_auc_avg_aucs.pkl')
fc_g_auc_dict = pd.read_pickle('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/treated_ranked/fc_g/rank_vs_auc_auc_dict.pkl')
viper_aucs = pd.read_pickle('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/treated_rank_filtered/viper_data/aucs.pkl')

plt.rcParams.update({'font.size': 13})

fig, ax = plt.subplots()
fig.set_figwidth(8.5)
fig.set_figheight(6.5)

fc_g_df = pd.DataFrame(fc_g_auc_dict)
fc_g_df['model_type'] = 'FC-G'
s_s_df = pd.DataFrame(s_s_auc_dict)
s_s_df['model_type'] = 'S-S'

df = pd.concat([s_s_df,fc_g_df])
df = pd.melt(df,id_vars=['model_type'],var_name=['cutoff'])
print(df)
viper_df = pd.DataFrame({str(c):viper_aucs[i] for i,c in enumerate(cutoffs)},index=['aucs']).T
viper_df = viper_df.reset_index(names='cutoffs')
print(viper_df)

def treat_plot(ax):
    palette = {'FC-G':'white','S-S':'lightgrey'}
    sns.boxplot(x='cutoff',y='value',hue='model_type',data=df,ax=ax,showfliers=False,palette=palette)
    sns.lineplot(x='cutoffs',y='aucs',data=viper_df,markers=True,color='red',ax=ax,label='viper')
    sns.scatterplot(x='cutoffs',y='aucs',data=viper_df,markers=True,color='red',ax=ax,edgecolor='w',linewidth=1.5,marker='o')
    #ax.set_ylim(0.3,0.8)
    ax.set_ylim(0.1,0.8)
    ax.set_ylabel('ROC AUC')
    ax.set_xlabel('Ranking Cutoff')
    ax.set_title('KO TF Gene Expression in Treated Sample\nRanking vs. Performance')
    ax.legend()
    ax.axhline(y=0.5,color='lightcoral',linestyle='--',zorder=0)

#treat_plot(ax)
#fig.savefig('treat_ranking.png')
"""
plt.plot(cutoffs,s_s_avg_aucs,label='S-S')
plt.plot(cutoffs,fc_g_avg_aucs,label='FC-G')
plt.plot(cutoffs,viper_aucs,label='viper')
plt.legend()
plt.ylabel('ROC AUC')
plt.xlabel('KO TF Gene Expression Ranking Cutoff in Treated Sample')

plt.savefig('treat_rank_avg_auc.png')
plt.clf()

sns.boxplot(data=pd.DataFrame(fc_g_auc_dict))
plt.ylabel("ROC AUC")
plt.xlabel('KO TF Gene Expression Ranking Cutoff in Treated Sample')
plt.title('FC-G')
plt.savefig('treat_rank_fc_g_aucs.png')
plt.clf()

sns.boxplot(data=pd.DataFrame(s_s_auc_dict))
plt.ylabel("ROC AUC")
plt.xlabel('KO TF Gene Expression Ranking Cutoff in Treated Sample')
plt.title('S-S')
plt.savefig('treat_rank_s_s_aucs.png')
"""
