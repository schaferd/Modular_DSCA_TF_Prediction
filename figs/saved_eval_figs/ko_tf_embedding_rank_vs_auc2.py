import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.insert(1,'/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/')
from get_roc_curve import getROC
from get_pert_roc_curves import getPertROC

SWARM_PLOT_SCATTER_SIZE=4.1
SWARM_PLOT_ALPHA=1
SWARM_PLOT_COLOR='k'
BOX_PLOT_OUTLINE_COLOR='slategray'
BOX_PLOT_WIDTH=0.5
BOX_PLOT_ORDER=0
BOX_PLOT_COLORS = {'boxprops':{'facecolor':'none', 'edgecolor':BOX_PLOT_OUTLINE_COLOR}, 'medianprops':{'color':BOX_PLOT_OUTLINE_COLOR},
    'whiskerprops':{'color':BOX_PLOT_OUTLINE_COLOR},
    'capprops':{'color':BOX_PLOT_OUTLINE_COLOR}}

base_path= '/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/'
s_s_path = [base_path+'shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_14.26.53/',base_path+'/shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/']
fc_g_path = [base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.56/',base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/']

s_s_activity_files = []
fc_g_activity_files = []
dorothea_s_s_activity_files = []
dorothea_fc_g_activity_files = []

for path in s_s_path:
    fold_paths = []
    fold_paths = fold_paths+[path+'/'+p for p in os.listdir(path) if 'fold' in p]
    act_path = []
    for p in fold_paths:
        act_path = act_path + [p+'/'+i for i in os.listdir(p) if 'ko_activities' in i]
        for p2 in act_path:
            s_s_activity_files = s_s_activity_files+[p2+'/'+f for f in os.listdir(p2) if 'knockout_diff_activities.csv' in f]
            dorothea_s_s_activity_files = dorothea_s_s_activity_files+[p2+'/'+f for f in os.listdir(p2) if ('diff_activities.csv' in f and 'knockout' not in f)]


for path in fc_g_path:
    fold_paths = []
    fold_paths = fold_paths+[path+'/'+p for p in os.listdir(path) if 'fold' in p]
    act_path = []
    for p in fold_paths:
        act_path = act_path + [p+'/'+i for i in os.listdir(p) if 'ko_activities' in i]
        for p2 in act_path:
            fc_g_activity_files = fc_g_activity_files+[p2+'/'+f for f in os.listdir(p2) if 'knockout_diff_activities.csv' in f]
            dorothea_fc_g_activity_files = dorothea_fc_g_activity_files+[p2+'/'+f for f in os.listdir(p2) if ('diff_activities.csv' in f and 'knockout' not in f)]



s_s_activities = [pd.read_csv(f,index_col=0) for f in s_s_activity_files]
fc_g_activities = [pd.read_csv(f,index_col=0) for f in fc_g_activity_files]
viper_activities = pd.read_csv('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/viper_data/inferred_TF_activities/VIPERdiff_activities.csv',sep='\t',index_col=0)
scenic_activities = pd.read_csv('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/viper_data/inferred_TF_activities/SCENICdiff_activities.csv',sep='\t',index_col=0)
id_to_kotf = pd.read_pickle('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/fold0_cycle0/ko_activities_cycle0_fold0/knocktf_sample_to_tf.pkl')
id_to_kotf = id_to_kotf.set_index('Sample_ID')
s_s_ensemble = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/shallow_shallow/ensemble_activities.csv',sep='\t',index_col=0)
fc_g_ensemble = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/fc_g/ensemble_activities.csv',sep='\t',index_col=0)

def consensus(act_list):
    tfs = set(act_list[0].columns)
    samples = set(act_list[0].index)
    for df in act_list:
        samples = samples.intersection(set(df.index))
        tfs = tfs.intersection(set(df.columns))
    tfs = list(tfs)
    samples = list(samples)
    tfs.sort()
    samples.sort()
    consensus_arr = []
    for df in act_list:
        df = df.loc[samples,tfs]
        df = df.sort_index(axis=0)
        df = ((df.T - df.T.mean())/df.T.std()).T
        consensus_arr.append(np.array([df.to_numpy()]))
    consensus_arr = np.vstack(consensus_arr)
    consensus = consensus_arr.mean(axis=0)
    consensus_df = pd.DataFrame(consensus, columns=tfs,index=samples)
    return consensus_df

viper_s_s_consensus = consensus([s_s_ensemble,viper_activities])

def get_ko_tf_df(rank_df):
    ko_tf_rank_dict = {}
    for i,row in rank_df.iterrows():
        tf = id_to_kotf.loc[i,'TF']
        tf_ranking = rank_df[tf].loc[i]
        ko_tf_rank_dict[i] = [tf_ranking]
    ko_tf_df = pd.DataFrame(ko_tf_rank_dict).T
    return ko_tf_df

#cutoffs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
cutoffs = [0]

def get_cutoff_samples(df, cutoff):
    return list(df[df[0] > cutoff].index)

def get_ranking_cutoff_aucs(activities,cutoffs):
    ranks = activities.rank(axis=1)/activities.shape[1]
    ko_df = get_ko_tf_df(ranks)
    aucs = []
    counts = []
    for c in cutoffs:
        cutoff_samples = get_cutoff_samples(ko_df,c)
        cutoff_act = activities.loc[cutoff_samples,:]
        cutoff_tfs = id_to_kotf.loc[cutoff_samples,'TF']
        if cutoff_act.shape[0] > 0:
            aucs.append(getROC(cutoff_act,cutoff_tfs,None).auc)
        else:
            aucs.append(0)
        counts.append(cutoff_act.shape[0])
    return aucs,counts

s_s_aucs = []
s_s_counts = []
for df in s_s_activities:
    aucs,counts = get_ranking_cutoff_aucs(df,cutoffs)
    s_s_aucs.append(aucs)
    s_s_counts.append(counts)

fc_g_aucs = []
fc_g_counts = []
for df in fc_g_activities:
    aucs,counts = get_ranking_cutoff_aucs(df,cutoffs)
    fc_g_aucs.append(aucs)
    fc_g_counts.append(counts)

def auc_list_to_dict(aucs,cutoffs):
    cutoff_dict = {}
    for l in aucs:
        for i,el in enumerate(l):
            if cutoffs[i] not in cutoff_dict:
                cutoff_dict[cutoffs[i]] = [el]
            else:
                cutoff_dict[cutoffs[i]].append(el)
    return cutoff_dict


s_s_auc_df = pd.DataFrame(auc_list_to_dict(s_s_aucs,cutoffs)).melt(value_vars=cutoffs,var_name='cutoff',value_name='AUC')
fc_g_auc_df = pd.DataFrame(auc_list_to_dict(fc_g_aucs,cutoffs)).melt(value_vars=cutoffs,var_name='cutoff',value_name='AUC')
s_s_auc_df['model'] = 'S-S'
fc_g_auc_df['model'] = 'FC-G'
ae_auc_df = pd.concat([s_s_auc_df,fc_g_auc_df],axis=0)

s_s_count_df = pd.DataFrame(auc_list_to_dict(s_s_counts,cutoffs)).melt(value_vars=cutoffs,var_name='cutoff',value_name='Counts')
fc_g_count_df = pd.DataFrame(auc_list_to_dict(fc_g_counts,cutoffs)).melt(value_vars=cutoffs,var_name='cutoff',value_name='Counts')
s_s_count_df['model'] = 'S-S'
fc_g_count_df['model'] = 'FC-G'
ae_count_df = pd.concat([s_s_count_df,fc_g_count_df],axis=0)

viper_aucs, viper_counts = get_ranking_cutoff_aucs(viper_activities,cutoffs)
scenic_aucs, scenic_counts = get_ranking_cutoff_aucs(scenic_activities,cutoffs)
s_s_e_aucs, s_s_e_counts = get_ranking_cutoff_aucs(s_s_ensemble,cutoffs)
fc_g_e_aucs, fc_g_e_counts = get_ranking_cutoff_aucs(fc_g_ensemble,cutoffs)
viper_s_s_aucs, viper_s_s_counts = get_ranking_cutoff_aucs(viper_s_s_consensus,cutoffs)

viper_auc_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':viper_aucs,'Counts':viper_counts})
scenic_auc_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':scenic_aucs,'Counts':scenic_counts})
s_s_e_auc_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':s_s_e_aucs,'Counts':s_s_e_counts})
fc_g_e_auc_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':fc_g_e_aucs,'Counts':fc_g_e_counts})
viper_s_s_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':viper_s_s_aucs,'Counts':viper_s_s_counts})

"""
ae_auc_df.to_pickle('ae_aucs_df.pkl')
viper_auc_df.to_pickle('viper_aucs_df.pkl')

viper_auc_df = pd.read_pickle('viper_aucs_df.pkl')
ae_auc_df = pd.read_pickle('ae_aucs_df.pkl')
"""

def count_plot(ax):
    #palette = {'FC-G':'white','S-S':'lightgrey'}
    #sns.boxplot(x='cutoff',y='Counts',hue='model',data=ae_auc_df,ax=ax,showfliers=False,palette=palette)

    sns.lineplot(x='cutoff',y='Counts',data=s_s_e_auc_df,markers=True,color='orange',ax=ax,label='S-S Ensemble')
    sns.scatterplot(x='cutoff',y='Counts',data=s_s_e_auc_df,markers=True,color='orange',ax=ax,edgecolor='w',linewidth=1.5,marker='o')

    sns.lineplot(x='cutoff',y='Counts',data=fc_g_e_auc_df,markers=True,color='blue',ax=ax,label='FC-G Ensemble')
    sns.scatterplot(x='cutoff',y='Counts',data=fc_g_e_auc_df,markers=True,color='blue',ax=ax,edgecolor='w',linewidth=1.5,marker='o')

    sns.lineplot(x='cutoff',y='Counts',data=viper_auc_df,markers=True,color='red',ax=ax,label='viper')
    sns.scatterplot(x='cutoff',y='Counts',data=viper_auc_df,markers=True,color='red',ax=ax,edgecolor='w',linewidth=1.5,marker='o')

    sns.lineplot(x='cutoff',y='Counts',data=scenic_auc_df,markers=True,color='black',ax=ax,label='AUCell')
    sns.scatterplot(x='cutoff',y='Counts',data=scenic_auc_df,markers=True,color='black',ax=ax,edgecolor='w',linewidth=1.5,marker='o')

    sns.lineplot(x='cutoff',y='Counts',data=viper_s_s_df,markers=True,color='slategrey',ax=ax,label='viper-S-S Consensus')
    sns.scatterplot(x='cutoff',y='Counts',data=viper_s_s_df,markers=True,color='slategrey',ax=ax,edgecolor='w',linewidth=1.5,marker='o')

    #ax.set_ylim(0.45,1)
    ax.set_ylabel('Number of Samples')
    ax.set_xlabel('Ranking Cutoff')
    ax.set_title('KO TF Embedding Ranking\nvs. Number of Samples')
    ax.legend()


def ko_tf_plot(ax):
    #palette = {'FC-G':'white','S-S':'lightgrey'}
    #sns.boxplot(x='cutoff',y='AUC',hue='model',data=ae_auc_df,ax=ax,showfliers=False,palette=palette)
    
    sns.lineplot(x='cutoff',y='AUC',data=s_s_e_auc_df,markers=True,color='orange',ax=ax,label='S-S Ensemble')
    sns.scatterplot(x='cutoff',y='AUC',data=s_s_e_auc_df,markers=True,color='orange',ax=ax,edgecolor='w',linewidth=1.5,marker='o')

    sns.lineplot(x='cutoff',y='AUC',data=fc_g_e_auc_df,markers=True,color='blue',ax=ax,label='FC-G Ensemble')
    sns.scatterplot(x='cutoff',y='AUC',data=fc_g_e_auc_df,markers=True,color='blue',ax=ax,edgecolor='w',linewidth=1.5,marker='o')

    sns.lineplot(x='cutoff',y='AUC',data=viper_auc_df,markers=True,color='red',ax=ax,label='viper')
    sns.scatterplot(x='cutoff',y='AUC',data=viper_auc_df,markers=True,color='red',ax=ax,edgecolor='w',linewidth=1.5,marker='o')

    sns.lineplot(x='cutoff',y='AUC',data=scenic_auc_df,markers=True,color='black',ax=ax,label='AUCell')
    sns.scatterplot(x='cutoff',y='AUC',data=scenic_auc_df,markers=True,color='black',ax=ax,edgecolor='w',linewidth=1.5,marker='o')

    sns.lineplot(x='cutoff',y='AUC',data=viper_s_s_df,markers=True,color='slategrey',ax=ax,label='viper-S-S Consensus')
    sns.scatterplot(x='cutoff',y='AUC',data=viper_s_s_df,markers=True,color='slategrey',ax=ax,edgecolor='w',linewidth=1.5,marker='o')

    ax.set_ylim(0.45,1)
    ax.set_ylabel('ROC AUC')
    ax.set_xlabel('Ranking Cutoff')
    ax.set_title('KO TF Embedding Ranking\nvs. Performance')
    ax.legend()
    ax.axhline(y=0.5,color='lightcoral',linestyle='--',zorder=0)

def all_sample_box_plot(ax):
    sns.boxplot(data=ae_auc_df[ae_auc_df['cutoff'] == 0],x='model',y='AUC',ax=ax,color='w',showfliers=False,zorder=BOX_PLOT_ORDER,width=BOX_PLOT_WIDTH,**BOX_PLOT_COLORS)
    sns.swarmplot(data=ae_auc_df[ae_auc_df['cutoff'] == 0],x='model',y='AUC',ax=ax,color=SWARM_PLOT_COLOR,size=SWARM_PLOT_SCATTER_SIZE,alpha=SWARM_PLOT_ALPHA)

    ax.axhline(y=viper_auc_df[viper_auc_df['cutoff'] == 0].loc[:,'AUC'][0],color='green',linestyle='--',zorder=0,label='viper')
    ax.axhline(y=scenic_auc_df[scenic_auc_df['cutoff'] == 0].loc[:,'AUC'][0],color='blue',linestyle='dashdot',zorder=0,label='AUCell')
    ax.axhline(y=viper_s_s_df[viper_s_s_df['cutoff'] == 0].loc[:,'AUC'][0],color='black',linestyle='dotted',zorder=0,label='viper-S-S Consensus')

    ax.set_ylim(0.48,0.8)
    #ax.set_ylabel('ROC AUC')
    ax.set_ylabel('')
    ax.set_xlabel('Model Type')
    ax.set_title('KnockTF Validation')
    #ax.legend(loc='lower left')
    ax.legend().set_visible(False)

"""
fig,ax = plt.subplots()
fig.set_figwidth(7)
fig.set_figheight(5)
ko_tf_plot(ax)
fig.savefig('embedding_ranking_vs_auc_line_plot.png')
#fig.savefig('embedding_ranking_vs_auc.png')

fig,ax = plt.subplots()
fig.set_figwidth(7)
fig.set_figheight(5)
count_plot(ax)
fig.savefig('embedding_ranking_vs_auc_counts.png')
"""

fig,ax = plt.subplots()
fig.set_figwidth(5)
fig.set_figheight(6)
all_sample_box_plot(ax)
fig.savefig('knocktf_boxplot.png')

###DOROTHEA BENCHMARK DATA
viper_activities = pd.read_csv('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/knn_inferred_tf_activities/VIPERdiff_activities.csv',sep='\t',index_col=0)
scenic_activities = pd.read_csv('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/knn_inferred_tf_activities/SCENICdiff_activities.csv',sep='\t',index_col=0)
dorothea_s_s_activities = [pd.read_csv(f,index_col=0)for f in dorothea_s_s_activity_files]
dorothea_fc_g_activities = [pd.read_csv(f,index_col=0)for f in dorothea_fc_g_activity_files]
dorothea_s_s_e = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/shallow_shallow/dorothea_ensemble_activities.csv',sep='\t',index_col=0) 
dorothea_fc_g_e = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/fc_g/dorothea_ensemble_activities.csv',sep='\t',index_col=0)
d_id_to_kotf = pd.read_pickle('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/knn_inferred_tf_activities/id_to_kotf.pkl')

viper_s_s = consensus([viper_activities,dorothea_s_s_e])


def get_aucs(activities):
    aucs = getPertROC(activities,d_id_to_kotf,None).auc
    return aucs

viper_auc = get_aucs(viper_activities.loc[viper_s_s.index,viper_s_s.columns])
scenic_auc = get_aucs(scenic_activities.loc[viper_s_s.index,viper_s_s.columns])
viper_s_s_auc = get_aucs(viper_s_s)
dorothea_s_s_e_auc = get_aucs(dorothea_s_s_e)
dorothea_s_s_aucs = [get_aucs(df.loc[viper_s_s.index,viper_s_s.columns]) for df in dorothea_s_s_activities]
dorothea_fc_g_aucs = [get_aucs(df.loc[viper_s_s.index,viper_s_s.columns]) for df in dorothea_fc_g_activities]

dorothea_ae_df = pd.DataFrame({'S-S':dorothea_s_s_aucs,'FC-G':dorothea_fc_g_aucs}).melt(value_vars=['S-S','FC-G'], var_name='model',value_name='AUC')

#print("scenic",scenic_auc)
#print("viper",viper_auc)
print("viper s s ",viper_s_s_auc)
#print("s s e",dorothea_s_s_e_auc)

def dorothea_all_sample_box_plot(ax):
    sns.boxplot(data=dorothea_ae_df,x='model',y='AUC',ax=ax,color='w',showfliers=False,zorder=BOX_PLOT_ORDER,width=BOX_PLOT_WIDTH,**BOX_PLOT_COLORS)
    sns.swarmplot(data=dorothea_ae_df,x='model',y='AUC',ax=ax,color=SWARM_PLOT_COLOR,size=SWARM_PLOT_SCATTER_SIZE,alpha=SWARM_PLOT_ALPHA)

    ax.axhline(y=viper_auc,color='forestgreen',linestyle='--',zorder=0,label='viper')
    ax.axhline(y=scenic_auc,color='mediumblue',linestyle='dashdot',zorder=0,label='AUCell')
    ax.axhline(y=viper_s_s_auc,color='purple',linestyle='dotted',zorder=0,label='viper-S-S Consensus')

    #ax.set_ylim(0.5,0.8)
    ax.set_ylabel('ROC AUC')
    ax.set_xlabel('Model Type')
    ax.set_title('Perturbation Validation')
    ax.legend(loc='lower left')

def plot_ko_results(left,right):
    dorothea_all_sample_box_plot(left)
    all_sample_box_plot(right)


fig,ax = plt.subplots(1,2,sharey=True)
fig.set_figwidth(8)
fig.set_figheight(5)
plot_ko_results(ax[0],ax[1])
#dorothea_all_sample_box_plot(ax)
fig.savefig('all_boxplot.png')





