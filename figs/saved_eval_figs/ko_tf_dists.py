import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


"""
viper_activity_df = pd.read_csv('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/viper_data/TF_activities/diff_activities.csv',sep='\t',index_col=0)
ae_activity_df = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/shallow_shallow/ensemble_activities.csv',sep='\t',index_col=0) 
id_to_kotf = pd.read_pickle('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/fold0_cycle0/ko_activities_cycle0_fold0/knocktf_sample_to_tf.pkl')
#id_to_kotf = id_to_kotf.set_index('Sample_ID')
"""
info_df = pd.read_csv('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/info.csv',sep='\t',index_col=0).set_index('sample_id')
print(info_df)

viper_activity_df = pd.read_csv('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/viper_data/TF_activities/VIPERdiff_activities.csv',sep='\t',index_col=0)
scenic_activity_df = pd.read_csv('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/viper_data/TF_activities/SCENICdiff_activities.csv',sep='\t',index_col=0)
id_to_kotf = pd.read_pickle('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/fold0_cycle0/ko_activities_cycle0_fold0/knocktf_sample_to_tf.pkl')
id_to_kotf = id_to_kotf.set_index('Sample_ID')
s_s_ensemble = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/shallow_shallow/ensemble_activities.csv',sep='\t',index_col=0)
fc_g_ensemble = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/fc_g/ensemble_activities.csv',sep='\t',index_col=0)

def consensus(act_list):
    tfs = set(act_list[0].columns)
    for df in act_list:
        tfs = tfs.intersection(set(df.columns))
    tfs = list(tfs)
    tfs.sort()
    consensus_arr = []
    for df in act_list:
        df = df.loc[:,tfs]
        df = df.sort_index(axis=0)
        df = ((df.T - df.T.mean())/df.T.std()).T
        consensus_arr.append(np.array([df.to_numpy()]))
    consensus_arr = np.vstack(consensus_arr)
    consensus = consensus_arr.mean(axis=0)
    consensus_df = pd.DataFrame(consensus, columns=tfs,index=act_list[0].index)
    return consensus_df

viper_s_s_consensus = consensus([s_s_ensemble,viper_activity_df])

#viper_activity_df = viper_activity_df.drop_duplicates()

viper_rank_df = viper_activity_df.rank(axis=1)
s_s_rank_df = s_s_ensemble.rank(axis=1)
fc_g_rank_df = fc_g_ensemble.rank(axis=1)
scenic_rank_df = scenic_activity_df.rank(axis=1)
v_s_s_rank_df = viper_s_s_consensus.rank(axis=1)

viper_rank_df = viper_rank_df/viper_rank_df.max()
s_s_rank_df = s_s_rank_df/s_s_rank_df.max()
fc_g_rank_df = fc_g_rank_df/fc_g_rank_df.max()
scenic_rank_df = scenic_rank_df/scenic_rank_df.max()
v_s_s_rank_df = v_s_s_rank_df/v_s_s_rank_df.max()

ko_tf_rank_dict = {}
ko_tf_activity_dict = {}
for i,row in viper_rank_df.iterrows():
    tf = id_to_kotf.loc[i,'TF']

    viper_ranking = viper_rank_df[tf].loc[i]
    s_s_ranking = s_s_rank_df[tf].loc[i]
    fc_g_ranking = fc_g_rank_df[tf].loc[i]
    scenic_ranking = scenic_rank_df[tf].loc[i]
    v_s_s_ranking = v_s_s_rank_df[tf].loc[i]

    ko_tf_rank_dict[i] = [s_s_ranking,fc_g_ranking,viper_ranking,scenic_ranking,v_s_s_ranking]

ko_tf_df = pd.DataFrame(ko_tf_rank_dict).T.rename(columns={0:'S-S',1:'FC-G',2:'viper',3:'AUCell',4:'viper-S-S Consensus'})


#melt_ko_tf_df = pd.melt(ko_tf_df,value_vars=ko_tf_df.columns,value_name='method',var_name='ranking')
#print(melt_ko_tf_df)
plt.clf()
fig,ax = plt.subplots(5,1,figsize=(5,9),sharey=True,sharex=True)
fig.subplots_adjust(left=0.1,bottom=0.2,right=0.9,top=0.8,wspace=0.35,hspace=0.4)
for i,method in enumerate(ko_tf_df.columns):
    a = ax[i]
    #sns.violinplot(x='method',y='ranking',data=melt_ko_tf_df,color='w')
    sns.distplot(ko_tf_df[method],kde=True,hist=True,ax=a,bins=6)
    a.set_title(method)
    a.set_xlabel('Ranking')
    #sns.swarmplot(x='method',y='ranking',data=melt_ko_tf_df)
fig.savefig('method_rank_boxplot.png')
raise ValueError()

"""
ko_tf_act_df = pd.DataFrame(ko_tf_activity_dict).T.rename(columns={0:'viper',1:'ae'})
print(ko_tf_df)
info_df = info_df.loc[ko_tf_df.index,:]
info_ko_tf_df = pd.concat([ko_tf_df,info_df],axis=1)
info_ko_tf_df['fc_rank'] = 1-info_ko_tf_df['fc_rank']
info_ko_tf_df['diff_rank'] = 1-info_ko_tf_df['diff_rank']
info_ko_tf_df['avg_rank'] = info_ko_tf_df.loc[:,info_df.columns].mean(axis=1)

fig,ax = plt.subplots()
fig.set_figwidth(8.5)
fig.set_figheight(6.5)
sns.scatterplot(x='viper',y='ae',hue='diff_rank',data=info_ko_tf_df,ax=ax)
fig.savefig('treated_rank_viper_vs_ae.png')

fig,ax = plt.subplots()
fig.set_figwidth(12.5)
fig.set_figheight(6.5)

print(ko_tf_df)
bottom_02 = ko_tf_df[ko_tf_df['ae'] < 0.2 ]
bottom_02 = list(bottom_02[bottom_02['viper'] >0.8].index)
print(bottom_02)
pd.to_pickle(bottom_02,'ae_bottom_02.pkl')
bottom_02_activities = viper_activity_df.loc[bottom_02,:].T.melt(value_vars=bottom_02,var_name='samples',value_name='activity')
print(bottom_02_activities)

ko_tf_df_bottom02 = ko_tf_act_df.loc[bottom_02,'viper'].reset_index(drop=False)
print(ko_tf_df_bottom02)

sns.boxplot(x='samples',y='activity',data=bottom_02_activities,color='w',ax=ax)
sns.swarmplot(x='samples',y='activity',data=bottom_02_activities,alpha=0.15,ax=ax)
sns.swarmplot(x='index',y='viper',data=ko_tf_df_bottom02,ax=ax)
fig.savefig('bottom_02.png')

tfs = []
for sample in ko_tf_df.index:
    tf = id_to_kotf.loc[sample,'TF']
    tfs.append(tf)

info_ko_tf_df['KO_TF'] = tfs
print(ko_tf_df)
"""

"""
fig,ax = plt.subplots()
fig.set_figwidth(40.5)
fig.set_figheight(6.5)
tf_rank_df_viper = pd.melt(viper_rank_df,value_vars=viper_rank_df.columns,value_name='ranking',var_name='TF')
sns.boxplot(x='TF',y='ranking',data=tf_rank_df_viper,color='w',ax=ax,showfliers=False)
sns.swarmplot(x='TF',y='ranking',data=tf_rank_df_viper,alpha=0.3,ax=ax,marker='.')
#sns.swarmplot(x='KO_TF',y='viper',hue='treated_rank',data=info_ko_tf_df)
sns.swarmplot(x='KO_TF',y='viper',data=info_ko_tf_df)
#sns.swarmplot(x='KO_TF',y='ae',data=info_ko_tf_df)
ax.set_title('VIPER TF Distributions')
ax.set_ylabel('Ranking')
ax.set_xlabel('TF')
fig.savefig('viper_tf_dists.png')

fig,ax = plt.subplots()
fig.set_figwidth(40.5)
fig.set_figheight(6.5)
tf_rank_df_ae = pd.melt(ae_rank_df,value_vars=ae_rank_df.columns,value_name='ranking',var_name='TF')
sns.boxplot(x='TF',y='ranking',data=tf_rank_df_ae,color='w',ax=ax,showfliers=False)
sns.swarmplot(x='TF',y='ranking',data=tf_rank_df_ae,alpha=0.3,ax=ax,marker='.')
#sns.swarmplot(x='KO_TF',y='ae',hue='treated_rank',data=info_ko_tf_df)
sns.swarmplot(x='KO_TF',y='ae',data=info_ko_tf_df)
#sns.swarmplot(x='KO_TF',y='ae',data=info_ko_tf_df)
ax.set_title('AE TF Distributions')
ax.set_ylabel('Ranking')
ax.set_xlabel('TF')
fig.savefig('ae_tf_dists.png')
"""
fig,ax = plt.subplots()
fig.set_figwidth(5)
fig.set_figheight(5)

plt.clf()
plt.scatter(ko_tf_df['viper'],ko_tf_df['ae'],marker='.')
plt.title('KO TF Rankings')
plt.xlabel('VIPER ranking')
plt.ylabel('AE ranking')
plt.savefig('viper_vs_ae_ranking.png',dpi=300)

plt.clf()
plt.hist(ko_tf_df['ae'],bins=30)
plt.ylim(0,40)
plt.title('AE KO TF rankings')
plt.savefig('ae_rank_hist.png')
plt.clf()

plt.hist(ko_tf_df['viper'],bins=30)
plt.ylim(0,40)
plt.title('VIPER KO TF Rankings')
plt.savefig('viper_rank_hist.png')

melt_ko_tf_df = pd.melt(ko_tf_df,value_vars=['viper','ae'],value_name='method',var_name='ranking')
print(melt_ko_tf_df)
plt.clf()
sns.boxplot(x='method',y='ranking',data=melt_ko_tf_df,color='w')
sns.swarmplot(x='method',y='ranking',data=melt_ko_tf_df)
plt.savefig('viper_vs_ae_boxplot.png')
plt.clf()

"""
print(ko_tf_df)

mean = ko_tf_df['viper'].mean()
print("viper mean",mean)
print("ae mean", ko_tf_df['ae'].mean())
print(ko_tf_df['viper'].to_list())
print(stats.kstest(ko_tf_df['viper'].to_list(),stats.uniform.cdf))
print(stats.kstest(ko_tf_df['ae'].to_list(),stats.uniform.cdf))

#ko_tf_df.to_pickle('ko_tf_df.pkl')





"""
