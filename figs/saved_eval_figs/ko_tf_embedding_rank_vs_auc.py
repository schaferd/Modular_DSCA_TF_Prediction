import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.insert(1,'/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/')
from get_roc_curve import getROC

base_path= '/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/'
s_s_path = [base_path+'shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_14.26.53/',base_path+'/shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/']
fc_g_path = [base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.56/',base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/']

s_s_activity_files = []
fc_g_activity_files = []

for path in s_s_path:
    fold_paths = []
    fold_paths = fold_paths+[path+'/'+p for p in os.listdir(path) if 'fold' in p]
    act_path = []
    for p in fold_paths:
        act_path = act_path + [p+'/'+i for i in os.listdir(p) if 'ko_activities' in i]
        for p2 in act_path:
            s_s_activity_files = s_s_activity_files+[p2+'/'+f for f in os.listdir(p2) if 'knockout_diff_activities.csv' in f]


for path in fc_g_path:
    fold_paths = []
    fold_paths = fold_paths+[path+'/'+p for p in os.listdir(path) if 'fold' in p]
    act_path = []
    for p in fold_paths:
        act_path = act_path + [p+'/'+i for i in os.listdir(p) if 'ko_activities' in i]
        for p2 in act_path:
            fc_g_activity_files = fc_g_activity_files+[p2+'/'+f for f in os.listdir(p2) if 'knockout_diff_activities.csv' in f]



s_s_activities = [pd.read_csv(f,index_col=0) for f in s_s_activity_files]
fc_g_activities = [pd.read_csv(f,index_col=0) for f in fc_g_activity_files]
viper_activities = pd.read_csv('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/viper_data/TF_activities/diff_activities.csv',sep='\t',index_col=0)
id_to_kotf = pd.read_pickle('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/fold0_cycle0/ko_activities_cycle0_fold0/knocktf_sample_to_tf.pkl')
id_to_kotf = id_to_kotf.set_index('Sample_ID')
s_s_ensemble = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/shallow_shallow/ensemble_activities.csv',sep='\t',index_col=0)
fc_g_ensemble = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/fc_g/ensemble_activities.csv',sep='\t',index_col=0)


s_s_ranks = [df.rank(axis=1)/df.shape[1] for df in s_s_activities]
fc_g_ranks = [df.rank(axis=1)/df.shape[1] for df in fc_g_activities]
viper_ranks = viper_activities.rank(axis=1)/viper_activities.shape[1]
s_s_e_ranks = s_s_ensemble.rank(axis=1)/s_s_ensemble.shape[1]
fc_g_e_ranks = fc_g_ensemble.rank(axis=1)/fc_g_ensemble.shape[1]

def get_ko_tf_df(rank_df):
    ko_tf_rank_dict = {}
    for i,row in rank_df.iterrows():
        tf = id_to_kotf.loc[i,'TF']
        tf_ranking = rank_df[tf].loc[i]
        ko_tf_rank_dict[i] = [tf_ranking]
    ko_tf_df = pd.DataFrame(ko_tf_rank_dict).T
    return ko_tf_df

s_s_ko_dfs = [get_ko_tf_df(df) for df in s_s_ranks]
fc_g_ko_dfs = [get_ko_tf_df(df) for df in fc_g_ranks]
viper_ko_df = get_ko_tf_df(viper_ranks)
s_s_e_ko_df = get_ko_tf_df(s_s_e_ranks)
fc_g_e_ko_df = get_ko_tf_df(fc_g_e_ranks)


cutoffs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

def get_cutoff_samples(df, cutoff):
    return list(df[df[0] > cutoff].index)

s_s_aucs = {}
fc_g_aucs = {}
viper_aucs = []
s_s_e_aucs = []
fc_g_e_aucs = []
for c in cutoffs:
    s_s_cutoff_samples = [get_cutoff_samples(df,c) for df in s_s_ko_dfs]
    fc_g_cutoff_samples = [get_cutoff_samples(df,c) for df in fc_g_ko_dfs]
    viper_cutoff_samples = get_cutoff_samples(viper_ko_df,c)
    s_s_e_cutoff_samples = get_cutoff_samples(s_s_e_ko_df,c)
    fc_g_e_cutoff_samples = get_cutoff_samples(fc_g_e_ko_df,c)
    
    s_s_cutoff_act = [df.loc[s_s_cutoff_samples[i],:] for i,df in enumerate(s_s_activities)]
    fc_g_cutoff_act = [df.loc[fc_g_cutoff_samples[i],:] for i,df in enumerate(fc_g_activities)]
    viper_cutoff_act = viper_activities.loc[viper_cutoff_samples,:]
    s_s_e_cutoff_act = s_s_ensemble.loc[s_s_e_cutoff_samples,:]
    fc_g_e_cutoff_act = fc_g_ensemble.loc[fc_g_e_cutoff_samples,:]

    s_s_cutoff_tfs = [id_to_kotf.loc[samples,'TF'] for samples in s_s_cutoff_samples]
    fc_g_cutoff_tfs = [id_to_kotf.loc[samples,'TF'] for samples in fc_g_cutoff_samples]
    viper_cutoff_tfs = id_to_kotf.loc[viper_cutoff_samples,'TF']
    s_s_e_cutoff_tfs = id_to_kotf.loc[s_s_e_cutoff_samples,'TF']
    fc_g_e_cutoff_tfs = id_to_kotf.loc[fc_g_e_cutoff_samples,'TF']

    auc_list = []
    for i,df in enumerate(s_s_cutoff_act):
        if df.shape[0] > 0:
            auc_list.append(getROC(df,s_s_cutoff_tfs[i],None).auc)
        else:
            auc_list.append(0)
    s_s_aucs[c] = auc_list

    auc_list = []
    for i,df in enumerate(fc_g_cutoff_act):
        if df.shape[0] > 0:
            auc_list.append(getROC(df,fc_g_cutoff_tfs[i],None).auc)
        else:
            auc_list.append(0)
    fc_g_aucs[c] = auc_list

    if viper_cutoff_act.shape[0] > 0:
        viper_aucs.append(getROC(viper_cutoff_act,viper_cutoff_tfs,None).auc)
    else:
        viper_aucs.append(0)
        
    if s_s_e_cutoff_act.shape[0] > 0:
        s_s_e_aucs.append(getROC(s_s_e_cutoff_act,s_s_e_cutoff_tfs,None).auc)
    else:
        s_s_e_aucs.append(0)

    if fc_g_e_cutoff_act.shape[0] > 0:
        fc_g_e_aucs.append(getROC(fc_g_e_cutoff_act,fc_g_e_cutoff_tfs,None).auc)
    else:
        fc_g_e_aucs.append(0)

s_s_auc_df = pd.DataFrame(s_s_aucs).melt(value_vars=cutoffs,var_name='cutoff',value_name='AUC')
fc_g_auc_df = pd.DataFrame(fc_g_aucs).melt(value_vars=cutoffs,var_name='cutoff',value_name='AUC')
s_s_auc_df['model'] = 'S-S'
fc_g_auc_df['model'] = 'FC-G'
ae_auc_df = pd.concat([s_s_auc_df,fc_g_auc_df],axis=0)

viper_auc_df = pd.DataFrame({'cutoff':[str(c) for c in cutoffs],'AUC':viper_aucs})
s_s_e_auc_df = pd.DataFrame({'cutoff':[str(c) for c in cutoffs],'AUC':s_s_e_aucs})
fc_g_e_auc_df = pd.DataFrame({'cutoff':[str(c) for c in cutoffs],'AUC':fc_g_e_aucs})


print("viper",viper_auc_df)
print("ae",ae_auc_df)
"""
ae_auc_df.to_pickle('ae_aucs_df.pkl')
viper_auc_df.to_pickle('viper_aucs_df.pkl')

viper_auc_df = pd.read_pickle('viper_aucs_df.pkl')
ae_auc_df = pd.read_pickle('ae_aucs_df.pkl')
"""


def ko_tf_plot(ax):
    palette = {'FC-G':'white','S-S':'lightgrey'}
    sns.boxplot(x='cutoff',y='AUC',hue='model',data=ae_auc_df,ax=ax,showfliers=False,palette=palette)

    sns.lineplot(x='cutoff',y='AUC',data=viper_auc_df,markers=True,color='red',ax=ax,label='viper')
    sns.scatterplot(x='cutoff',y='AUC',data=viper_auc_df,markers=True,color='red',ax=ax,edgecolor='w',linewidth=1.5,marker='o')

    sns.lineplot(x='cutoff',y='AUC',data=s_s_e_auc_df,markers=True,color='purple',ax=ax,label='S-S Ensemble')
    sns.scatterplot(x='cutoff',y='AUC',data=s_s_e_auc_df,markers=True,color='purple',ax=ax,edgecolor='w',linewidth=1.5,marker='o')

    sns.lineplot(x='cutoff',y='AUC',data=fc_g_e_auc_df,markers=True,color='blue',ax=ax,label='FC-G Ensemble')
    sns.scatterplot(x='cutoff',y='AUC',data=fc_g_e_auc_df,markers=True,color='purple',ax=ax,edgecolor='w',linewidth=1.5,marker='o')

    ax.set_ylim(0.4,1)
    ax.set_ylabel('ROC AUC')
    ax.set_xlabel('Ranking Cutoff')
    ax.set_title('KO TF Embedding\nRanking vs. Performance')
    ax.legend()
    ax.axhline(y=0.5,color='lightcoral',linestyle='--',zorder=0)

fig,ax = plt.subplots()
fig.set_figwidth(7)
fig.set_figheight(5)
ko_tf_plot(ax)
fig.savefig('embedding_ranking_vs_auc.png')
