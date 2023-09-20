import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy import stats
from pyensembl import EnsemblRelease
from sklearn.linear_model import LinearRegression
ensembl_data = EnsemblRelease(78)
from load_attribution_scores import get_attr_scores_one_run, ensembl_to_gene_name, get_attr_scores_avg_runs
sys.path.insert(1,'/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/')
from get_roc_curve import getROC
from get_pert_roc_curves import getPertROC

full_path = "/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/figs/saved_eval_figs/attribution_scores/"

attr_df = get_attr_scores_one_run()

tf_sums = attr_df.abs().sum(axis=0)
#tf_sums = {tf:tf_sums[tf] for tf in tf_sums.index}
tf_sum_names = tf_sums.index
tf_attr = np.array(list(tf_sums))

filtered_pkn = pd.read_csv(full_path+'filtered_pkn.tsv',sep='\t',index_col=0)
tfs_in_pkn = np.unique(filtered_pkn['tf'])
tfs_rel = [len(filtered_pkn[filtered_pkn['tf'] == tf]) for tf in tfs_in_pkn]

base_path= '/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/'
s_s_path = [base_path+'shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_14.26.53/',base_path+'/shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/']
fc_g_path = [base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.56/',base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/']

s_s_activity_files = []
fc_g_activity_files = []
dorothea_s_s_activity_files = []
dorothea_fc_g_activity_files = []
knocktf_recon_corr_s_s = []
knocktf_recon_corr_fc_g = []
dorothea_recon_corr_s_s = []
dorothea_recon_corr_fc_g = []

for path in s_s_path:
    fold_paths = []
    fold_paths = fold_paths+[path+'/'+p for p in os.listdir(path) if 'fold' in p]
    act_path = []
    for p in fold_paths:
        act_path = act_path + [p+'/'+i for i in os.listdir(p) if 'ko_activities' in i]
        for p2 in act_path:
            s_s_activity_files = s_s_activity_files+[p2+'/'+f for f in os.listdir(p2) if 'knockout_diff_activities.csv' in f]
            dorothea_s_s_activity_files = dorothea_s_s_activity_files+[p2+'/'+f for f in os.listdir(p2) if ('diff_activities.csv' in f and 'knockout' not in f)]
            dorothea_recon_corr_s_s = dorothea_recon_corr_s_s+[p2+'/'+f for f in os.listdir(p2) if 'dorothea_recon_corr' in f]
            knocktf_recon_corr_s_s = knocktf_recon_corr_s_s+[p2+'/'+f for f in os.listdir(p2) if 'knocktf_recon_corr' in f]


for path in fc_g_path:
    fold_paths = []
    fold_paths = fold_paths+[path+'/'+p for p in os.listdir(path) if 'fold' in p]
    act_path = []
    for p in fold_paths:
        act_path = act_path + [p+'/'+i for i in os.listdir(p) if 'ko_activities' in i]
        for p2 in act_path:
            fc_g_activity_files = fc_g_activity_files+[p2+'/'+f for f in os.listdir(p2) if 'knockout_diff_activities.csv' in f]
            dorothea_fc_g_activity_files = dorothea_fc_g_activity_files+[p2+'/'+f for f in os.listdir(p2) if ('diff_activities.csv' in f and 'knockout' not in f)]
            dorothea_recon_corr_fc_g = dorothea_recon_corr_fc_g+[p2+'/'+f for f in os.listdir(p2) if 'dorothea_recon_corr' in f]
            knocktf_recon_corr_fc_g = knocktf_recon_corr_fc_g+[p2+'/'+f for f in os.listdir(p2) if 'knocktf_recon_corr' in f]


knocktf_recon_corr_s_s = [pd.read_pickle(f) for f in knocktf_recon_corr_s_s]
knocktf_recon_corr_fc_g = [pd.read_pickle(f) for f in knocktf_recon_corr_fc_g]
dorothea_recon_corr_s_s = [pd.read_pickle(f) for f in dorothea_recon_corr_s_s]
dorothea_recon_corr_fc_g = [pd.read_pickle(f) for f in dorothea_recon_corr_fc_g]

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


def get_recon_corr_of_sample(dfs,id_to_kotf,recon_corrs):
    recon_corr_list = []
    ko_tf_list = [id_to_kotf.loc[i,'TF'] for i in dfs[0].index]
    for c in recon_corrs:
        recon_corr_list.append(np.array([c[i] for i in dfs[0].index]))

    recon_corr_list = np.array(recon_corr_list)
    recon_corr_mean = recon_corr_list.mean(axis=0)

    
    return recon_corr_mean, ko_tf_list

def get_variance_of_sample_across_models(dfs,id_to_kotf):
    np_dfs = np.array([df.to_numpy() for df in dfs])
    np_dfs = ((np_dfs - np_dfs.mean(axis=0))**2).sum(axis=0)/(np_dfs.shape[0])
    avg_var = np_dfs.mean(axis=1)
    return np.array(avg_var), np.array([id_to_kotf.loc[i,'TF'] for i in dfs[0].index])

def get_variance_of_kotf_across_models(dfs,id_to_kotf):
    np_dfs = np.array([df.to_numpy() for df in dfs])
    np_dfs = ((np_dfs - np_dfs.mean(axis=0))**2).sum(axis=0)/(np_dfs.shape[0])
    kotfs = np.array([id_to_kotf.loc[i,'TF'] for i in dfs[0].index])
    kotf_vars = []
    for i,row in enumerate(np_dfs):
        kotf = kotfs[i]
        kotf_vars.append(row[list(dfs[0].columns).index(kotf)])

    return np.array(kotf_vars), kotfs


"""
def get_cutoff_samples_corr(df,cutoff,corrs,kotfs):
    tfs = kotfs[corrs > cutoff]
    ko_tfs = np.array([id_to_kotf.loc[i,'TF'] for i in df.index])
    filtered_df = df.loc[np.isin(ko_tfs,tfs),:]
    return filtered_df, ko_tfs[np.isin(ko_tfs,tfs)]

def get_cutoff_samples_var(df,cutoff,avg_vars,tf_names,id_to_kotf):
    tfs = tf_names[avg_vars > cutoff]
    ko_tfs = np.array([id_to_kotf.loc[i,'TF'] for i in df.index])
    filtered_df = df.loc[np.isin(ko_tfs,tfs),:]
    return filtered_df, ko_tfs[np.isin(ko_tfs,tfs)]

def get_cutoff_samples_attr(df,cutoff,id_to_kotf):
    tfs = tf_names[tf_attr > cutoff]
    ko_tfs = np.array([id_to_kotf.loc[i,'TF'] for i in df.index])
    filtered_df = df.loc[np.isin(ko_tfs,tfs),:]
    return filtered_df, ko_tfs[np.isin(ko_tfs,tfs)]

def get_cutoff_samples_rel(df,cutoff,id_to_kotf):
    tfs = tfs_in_pkn[tfs_rel > cutoff]
    ko_tfs = np.array([id_to_kotf.loc[i,'TF'] for i in df.index])
    filtered_df = df.loc[np.isin(ko_tfs,tfs),:]
    return filtered_df, ko_tfs[np.isin(ko_tfs,tfs)]
"""

def get_cutoff_samples(df,cutoff,values,names,id_to_kotf):
    names = np.array(names)
    tfs = names[values > cutoff]
    ko_tfs = np.array([id_to_kotf.loc[i,'TF'] for i in df.index])
    filtered_df = df.loc[np.isin(ko_tfs,tfs),:]
    return filtered_df, ko_tfs[np.isin(ko_tfs,tfs)]

def get_cutoff_samples_fractional(df,cutoff,values,names,id_to_kotf):
    names = np.array(names)
    sorted_ = np.argsort(values)/len(values)
    tfs = names[sorted_ > cutoff]
    ko_tfs = np.array([id_to_kotf.loc[i,'TF'] for i in df.index])
    filtered_df = df.loc[np.isin(ko_tfs,tfs),:]
    return filtered_df, ko_tfs[np.isin(ko_tfs,tfs)]

def get_ranking_cutoff_aucs(activities,cutoffs,id_to_kotf,values,names):
    aucs = []
    counts = []
    for c in cutoffs:
        cutoff_samples, cutoff_tfs = get_cutoff_samples_fractional(activities,c,values, names,id_to_kotf)
        if cutoff_samples.shape[0] > 0:
            aucs.append(getROC(cutoff_samples,cutoff_tfs,None).auc)
        else:
            aucs.append(0)
        counts.append(cutoff_samples.shape[0])
    return aucs,counts

def get_ranking_fractional_cutoff_aucs(activities,cutoffs,id_to_kotf,values,names):
    aucs = []
    counts = []
    for c in cutoffs:
        cutoff_samples, cutoff_tfs = get_cutoff_samples_fractional(activities,c,values, names,id_to_kotf)
        if cutoff_samples.shape[0] > 0:
            aucs.append(getROC(cutoff_samples,cutoff_tfs,None).auc)
        else:
            aucs.append(0)
        counts.append(cutoff_samples.shape[0])
    return aucs,counts

def auc_list_to_dict(aucs,cutoffs):
    cutoff_dict = {}
    for l in aucs:
        for i,el in enumerate(l):
            if cutoffs[i] not in cutoff_dict:
                cutoff_dict[cutoffs[i]] = [el]
            else:
                cutoff_dict[cutoffs[i]].append(el)
    return cutoff_dict

#cutoffs = np.arange(0,1.2,0.05)
#cutoffs = np.arange(0,0.9,0.05)
#cutoffs = np.arange(0,500,50)
#cutoffs = np.arange(0.005,0.05,0.005)
#cutoffs = np.arange(-0.05,0.30,0.05)
cutoffs = np.arange(0,1,0.1)

def knocktf_plot():
    s_s_values, s_s_names = get_recon_corr_of_sample(s_s_activities,id_to_kotf,knocktf_recon_corr_s_s)
    fc_g_values,fc_g_names = get_recon_corr_of_sample(s_s_activities,id_to_kotf,knocktf_recon_corr_fc_g)

    """
    s_s_values, s_s_names = get_variance_of_kotf_across_models(s_s_activities,id_to_kotf)
    fc_g_values, fc_g_names = get_variance_of_kotf_across_models(fc_g_activities,id_to_kotf)
    """
    #s_s_values, s_s_names = get_variance_of_sample_across_models(s_s_activities,id_to_kotf)
    #fc_g_values, fc_g_names = get_variance_of_sample_across_models(fc_g_activities,id_to_kotf)

    #s_s_values, s_s_names = tf_attr,tf_sum_names 
    #fc_g_values, fc_g_names = tf_attr,tf_sum_names

    #s_s_values, s_s_names = tfs_rel,tfs_in_pkn 
    #fc_g_values, fc_g_names = tfs_rel,tfs_in_pkn 
    
    viper_s_s_consensus = consensus([s_s_ensemble,viper_activities])
    viper_fc_g_consensus = consensus([fc_g_ensemble,viper_activities])
    fc_g_s_s_consensus = consensus([fc_g_ensemble,s_s_ensemble])
    scenic_s_s_consensus = consensus([scenic_activities,s_s_ensemble])
    scenic_fc_g_consensus = consensus([scenic_activities,fc_g_ensemble])

    s_s_aucs = []
    s_s_counts = []
    for df in s_s_activities:
        aucs,counts = get_ranking_cutoff_aucs(df,cutoffs,id_to_kotf,s_s_values,s_s_names)
        s_s_aucs.append(aucs)
        s_s_counts.append(counts)

    fc_g_aucs = []
    fc_g_counts = []
    for df in fc_g_activities:
        aucs,counts = get_ranking_cutoff_aucs(df,cutoffs,id_to_kotf,fc_g_values,fc_g_names)
        fc_g_aucs.append(aucs)
        fc_g_counts.append(counts)

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

    viper_aucs, viper_counts = get_ranking_cutoff_aucs(viper_activities,cutoffs,id_to_kotf,s_s_values,s_s_names)
    scenic_aucs, scenic_counts = get_ranking_cutoff_aucs(scenic_activities,cutoffs,id_to_kotf,s_s_values,s_s_names)
    s_s_e_aucs, s_s_e_counts = get_ranking_cutoff_aucs(s_s_ensemble,cutoffs,id_to_kotf,s_s_values,s_s_names)
    fc_g_e_aucs, fc_g_e_counts = get_ranking_cutoff_aucs(fc_g_ensemble,cutoffs,id_to_kotf,fc_g_values,fc_g_names)
    viper_s_s_aucs, viper_s_s_counts = get_ranking_cutoff_aucs(viper_s_s_consensus,cutoffs,id_to_kotf,s_s_values,s_s_names)
    viper_fc_g_aucs, viper_fc_g_counts = get_ranking_cutoff_aucs(viper_fc_g_consensus,cutoffs,id_to_kotf,fc_g_values,fc_g_names)
    fc_g_s_s_aucs, fc_g_s_s_counts = get_ranking_cutoff_aucs(fc_g_s_s_consensus,cutoffs,id_to_kotf,fc_g_values,fc_g_names)
    scenic_fc_g_aucs, scenic_fc_g_counts = get_ranking_cutoff_aucs(scenic_fc_g_consensus,cutoffs,id_to_kotf,fc_g_values,fc_g_names)
    scenic_s_s_aucs, scenic_s_s_counts = get_ranking_cutoff_aucs(scenic_s_s_consensus,cutoffs,id_to_kotf,s_s_values,s_s_names)


    viper_auc_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':viper_aucs,'Counts':viper_counts})
    scenic_auc_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':scenic_aucs,'Counts':scenic_counts})
    s_s_e_auc_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':s_s_e_aucs,'Counts':s_s_e_counts})
    fc_g_e_auc_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':fc_g_e_aucs,'Counts':fc_g_e_counts})
    viper_s_s_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':viper_s_s_aucs,'Counts':viper_s_s_counts})
    viper_fc_g_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':viper_fc_g_aucs,'Counts':viper_fc_g_counts})
    fc_g_s_s_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':fc_g_s_s_aucs,'Counts':fc_g_s_s_counts})
    scenic_fc_g_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':scenic_fc_g_aucs,'Counts':scenic_fc_g_counts})
    scenic_s_s_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':scenic_s_s_aucs,'Counts':scenic_s_s_counts})

    return {'s_s_e_auc_df':s_s_e_auc_df,'fc_g_e_auc_df':fc_g_e_auc_df,'viper_auc_df':viper_auc_df,'scenic_auc_df': scenic_auc_df,'viper_s_s_df': viper_s_s_df,'viper_fc_g_df': viper_fc_g_df,'fc_g_s_s_df': fc_g_s_s_df,'scenic_fc_g_df':scenic_fc_g_df, 'scenic_s_s_df':scenic_s_s_df}

def ko_tf_plot(ax,title,s_s_e_auc_df,fc_g_e_auc_df,viper_auc_df,scenic_auc_df,viper_s_s_df,viper_fc_g_df,legend=True):
    #palette = {'FC-G':'white','S-S':'lightgrey'}
    #sns.boxplot(x='cutoff',y='AUC',hue='model',data=ae_auc_df,ax=ax,showfliers=False,palette=palette)
    
    sns.lineplot(x='cutoff',y='AUC',data=s_s_e_auc_df,markers=True,color='orange',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=s_s_e_auc_df,markers=True,color='orange',ax=ax,edgecolor='w',linewidth=1.5,marker='o',label='S-S Ensemble')

    sns.lineplot(x='cutoff',y='AUC',data=fc_g_e_auc_df,markers=True,color='blue',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=fc_g_e_auc_df,markers=True,color='blue',ax=ax,marker='v',label='FC-G Ensemble')

    sns.lineplot(x='cutoff',y='AUC',data=viper_auc_df,markers=True,color='red',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=viper_auc_df,markers=True,color='red',ax=ax,edgecolor='w',linewidth=1.5,marker='x',label='viper')

    sns.lineplot(x='cutoff',y='AUC',data=scenic_auc_df,markers=True,color='black',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=scenic_auc_df,markers=True,color='black',ax=ax,marker='d',label='AUCell')

    sns.lineplot(x='cutoff',y='AUC',data=viper_s_s_df,markers=True,color='slategrey',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=viper_s_s_df,markers=True,color='slategrey',ax=ax,marker='D',label='viper-S-S Consensus')

    sns.lineplot(x='cutoff',y='AUC',data=viper_fc_g_df,markers=True,color='purple',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=viper_fc_g_df,markers=True,color='purple',ax=ax,marker='s',label='viper-FC-G Consensus')

    #ax.set_ylim(0.45,1)
    ax.set_ylabel('ROC AUC')
    #ax.set_xlabel('Attribution Sum Ranking Cutoff')
    #ax.set_xlabel('TF Interactions Ranking Cutoff')
    #ax.set_xlabel('Sample Variance Cutoff Ranking')
    #ax.set_xlabel('KO TF Variance Cutoff Ranking')
    #ax.set_xlabel('Sample Reconstruction Correlation Ranking')
    ax.set_xlabel('Ranking Cutoff')

    #ax.set_title('Attribution Score Sum Ranking\nvs. KnockTF Performance')
    #ax.set_title('Attribution Score Sum\nvs. DoRothEA Benchmark Performance')
    #ax.set_title('Number of TF interactions Ranking\nvs. DoRothEA Benchmark Performance')
    #ax.set_title('Number of TF interactions\nvs. KnockTF Performance')
    #ax.set_title('Sample Variance Ranking\nvs. DoRothEA Benchmark Performance')
    #ax.set_title('Sample Variance Ranking\nvs. KnockTF Performance')
    #ax.set_title('KO TF Variance Ranking\nvs. DoRothEA Benchmark Performance')
    #ax.set_title('KO TF Variance Ranking\nvs. KnockTF Performance')
    #ax.set_title('Sample Reconstruction Correlation Ranking\nvs. DoRothEA Benchmark Performance')
    #ax.set_title('Sample Reconstruction Correlation Ranking\nvs. KnockTF Performance')
    #ax.set_title(title)

    if legend:
        ax.legend()
    else:
        ax.legend().set_visible(False)

    ax.axhline(y=0.5,color='lightcoral',linestyle='--',zorder=0)

def fc_g_s_s_consensus_plot(ax,title,fc_g_e_auc_df,s_s_e_auc_df,fc_g_s_s_df,ymin=0.45,ymax=1,legend=True):
    sns.lineplot(x='cutoff',y='AUC',data=fc_g_e_auc_df,markers=True,color='blue',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=fc_g_e_auc_df,markers=True,color='blue',ax=ax,marker='v',label='FC-G Ensemble')

    sns.lineplot(x='cutoff',y='AUC',data=s_s_e_auc_df,markers=True,color='orange',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=s_s_e_auc_df,markers=True,color='orange',ax=ax,edgecolor='w',linewidth=1.5,marker='o',label='S-S Ensemble')

    sns.lineplot(x='cutoff',y='AUC',data=fc_g_s_s_df,markers=True,color='deeppink',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=fc_g_s_s_df,markers=True,color='deeppink',ax=ax,marker='s',label='S-S-FC-G Consensus')

    ax.set_ylim(ymin,ymax)
    ax.set_ylabel('ROC AUC')
    ax.set_xlabel('Reconstruction Correlation\nRanking Cutoff')

    #ax.set_title(title)

    if legend:
        ax.legend()
    else:
        ax.legend().set_visible(False)

    ax.axhline(y=0.5,color='lightcoral',linestyle='--',zorder=0)

def fc_g_s_s_consensus_hists(ax,title,fc_g_e_auc_df,s_s_e_auc_df,fc_g_s_s_df):
    names = ['S-S','FC-G','S-S-FC-G']
    lines = [s_s_e_auc_df,fc_g_e_auc_df,fc_g_s_s_df]
    derivs = quantify_method_increase(ax,lines,names)
    colors = ['orange','blue','deeppink']
    derivs = pd.DataFrame(derivs)
    derivs = pd.melt(derivs,value_vars=derivs.columns,var_name='Method',value_name='Slope')
    sns.kdeplot(data=derivs,x='Slope',hue='Method',fill=True,palette=colors,legend=True,ax=ax)

def s_s_scenic_consensus_plot(ax,title,s_s_e_auc_df,scenic_auc_df,scenic_s_s_df,ymin=0.45,ymax=1,legend=False):
    sns.lineplot(x='cutoff',y='AUC',data=scenic_auc_df,markers=True,color='black',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=scenic_auc_df,markers=True,color='black',ax=ax,marker='d',label='AUCell')

    sns.lineplot(x='cutoff',y='AUC',data=s_s_e_auc_df,markers=True,color='orange',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=s_s_e_auc_df,markers=True,color='orange',ax=ax,edgecolor='w',linewidth=1.5,marker='o',label='S-S Ensemble')

    sns.lineplot(x='cutoff',y='AUC',data=scenic_s_s_df,markers=True,color='olive',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=scenic_s_s_df,markers=True,color='olive',ax=ax,marker='s',label='AUCell-S-S Consensus')

    ax.set_ylim(ymin,ymax)
    ax.set_ylabel('ROC AUC')
    ax.set_xlabel('Reconstruction Correlation\nRanking Cutoff')

    #ax.set_title(title)

    if legend:
        ax.legend()
    else:
        ax.legend().set_visible(False)

    ax.axhline(y=0.5,color='lightcoral',linestyle='--',zorder=0)

def s_s_scenic_consensus_hists(ax,title,s_s_e_auc_df,scenic_auc_df,scenic_s_s_df):
    names = ['S-S','AUCell','S-S-AUCell']
    lines = [s_s_e_auc_df,scenic_auc_df,scenic_s_s_df]
    derivs = quantify_method_increase(ax,lines,names)
    colors = ['orange','black','olive']
    derivs = pd.DataFrame(derivs)
    derivs = pd.melt(derivs,value_vars=derivs.columns,var_name='Method',value_name='Slope')
    sns.kdeplot(data=derivs,x='Slope',hue='Method',fill=True,palette=colors,legend=True,ax=ax)

def fc_g_scenic_consensus_plot(ax,title,fc_g_e_auc_df,scenic_auc_df,scenic_fc_g_df,ymin=0.45,ymax=1,legend=False):
    sns.lineplot(x='cutoff',y='AUC',data=scenic_auc_df,markers=True,color='black',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=scenic_auc_df,markers=True,color='black',ax=ax,marker='d',label='AUCell')

    sns.lineplot(x='cutoff',y='AUC',data=fc_g_e_auc_df,markers=True,color='blue',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=fc_g_e_auc_df,markers=True,color='blue',ax=ax,edgecolor='w',linewidth=1.5,marker='v',label='FC-G Ensemble')

    sns.lineplot(x='cutoff',y='AUC',data=scenic_fc_g_df,markers=True,color='teal',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=scenic_fc_g_df,markers=True,color='teal',ax=ax,marker='s',label='AUCell-FC-G Consensus')

    #ax.set_ylim(0.45,1)
    ax.set_ylim(ymin,ymax)
    ax.set_ylabel('ROC AUC')
    ax.set_xlabel('Reconstruction Correlation\nRanking Cutoff')

    #ax.set_title(title)

    if legend:
        ax.legend()
    else:
        ax.legend().set_visible(False)

    ax.axhline(y=0.5,color='lightcoral',linestyle='--',zorder=0)

def fc_g_scenic_consensus_hists(ax,title,fc_g_e_auc_df,scenic_auc_df,scenic_fc_g_df):
    names = ['FC-G','AUCell','FC-G-AUCell']
    lines = [fc_g_e_auc_df,scenic_auc_df,scenic_fc_g_df]
    derivs = quantify_method_increase(ax,lines,names)
    colors = ['blue','black','teal']
    derivs = pd.DataFrame(derivs)
    derivs = pd.melt(derivs,value_vars=derivs.columns,var_name='Method',value_name='Slope')
    sns.kdeplot(data=derivs,x='Slope',hue='Method',fill=True,palette=colors,legend=True,ax=ax)

def fc_g_consensus_plot(ax,title,fc_g_e_auc_df,viper_auc_df,viper_fc_g_df,ymin=0.45,ymax=1,legend=True):
    sns.lineplot(x='cutoff',y='AUC',data=fc_g_e_auc_df,markers=True,color='blue',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=fc_g_e_auc_df,markers=True,color='blue',ax=ax,marker='v',label='FC-G Ensemble')

    sns.lineplot(x='cutoff',y='AUC',data=viper_auc_df,markers=True,color='red',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=viper_auc_df,markers=True,color='red',ax=ax,edgecolor='w',linewidth=1.5,marker='x',label='viper')

    sns.lineplot(x='cutoff',y='AUC',data=viper_fc_g_df,markers=True,color='purple',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=viper_fc_g_df,markers=True,color='purple',ax=ax,marker='s',label='viper-FC-G Consensus')

    ax.set_ylim(ymin,ymax)
    ax.set_ylabel('ROC AUC')
    ax.set_xlabel('Reconstruction Correlation\nRanking Cutoff')

    #ax.set_title(title)

    if legend:
        ax.legend()
    else:
        ax.legend().set_visible(False)

    ax.axhline(y=0.5,color='lightcoral',linestyle='--',zorder=0)

def fc_g_viper_consensus_hists(ax,title,fc_g_e_auc_df,viper_auc_df,viper_fc_g_df):
    names = ['FC-G','viper','FC-G-viper']
    lines = [fc_g_e_auc_df,viper_auc_df,viper_fc_g_df]
    derivs = quantify_method_increase(ax,lines,names)
    colors = ['blue','red','purple']
    derivs = pd.DataFrame(derivs)
    derivs = pd.melt(derivs,value_vars=derivs.columns,var_name='Method',value_name='Slope')
    sns.kdeplot(data=derivs,x='Slope',hue='Method',fill=True,palette=colors,legend=True,ax=ax)

def s_s_consensus_plot(ax,title,s_s_e_auc_df,viper_auc_df,viper_s_s_df,ymin=0.45,ymax=1,legend=True):
    lines = [s_s_e_auc_df,viper_auc_df,viper_s_s_df]
    names = ['s_s_e','viper','viper_s_s']
    quantify_method_increase(ax,lines,names)

    sns.lineplot(x='cutoff',y='AUC',data=s_s_e_auc_df,markers=True,color='orange',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=s_s_e_auc_df,markers=True,color='orange',ax=ax,edgecolor='w',linewidth=1.5,marker='o',label='S-S Ensemble')

    sns.lineplot(x='cutoff',y='AUC',data=viper_auc_df,markers=True,color='red',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=viper_auc_df,markers=True,color='red',ax=ax,edgecolor='w',linewidth=1.5,marker='x',label='viper')

    sns.lineplot(x='cutoff',y='AUC',data=viper_s_s_df,markers=True,color='slategrey',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=viper_s_s_df,markers=True,color='slategrey',ax=ax,marker='D',label='viper-S-S Consensus')

    #ax.set_ylim(0.45,1)
    ax.set_ylim(ymin,ymax)
    ax.set_ylabel('ROC AUC')
    ax.set_xlabel('Reconstruction Correlation\nRanking Cutoff')

    #ax.set_title(title)

    if legend:
        ax.legend()
    else:
        ax.legend().set_visible(False)

    ax.axhline(y=0.5,color='lightcoral',linestyle='--',zorder=0)

def s_s_viper_consensus_hists(ax,title,s_s_e_auc_df,viper_auc_df,viper_s_s_df):
    names = ['S-S','viper','S-S-viper']
    lines = [s_s_e_auc_df,viper_auc_df,viper_s_s_df]
    derivs = quantify_method_increase(ax,lines,names)
    colors = ['orange','red','slategrey']
    derivs = pd.DataFrame(derivs)
    derivs = pd.melt(derivs,value_vars=derivs.columns,var_name='Method',value_name='Slope')
    sns.kdeplot(data=derivs,x='Slope',hue='Method',fill=True,palette=colors,legend=True,ax=ax)

def quantify_method_increase(ax,lines,names):
    derivs = {}
    for i,line in enumerate(lines):
        reg = LinearRegression().fit(np.array(line['cutoff']).reshape(-1, 1),line['AUC'])
        error = reg.score(np.array(line['cutoff']).reshape(-1, 1),line['AUC'])
        corr,pval = stats.pearsonr(line['cutoff'],line['AUC'])
        deriv = []
        cutoffs = np.array(line['cutoff'])
        aucs = np.array(line['AUC'])
        for j,auc in enumerate(aucs):
            if j != 0:
                deriv.append((aucs[j]-aucs[j-1])/(cutoffs[j]-cutoffs[j-1]))
        derivs[names[i]] = deriv
        #print(names[i])
        n = len(line['AUC'])
        #print(derivs[names[i]])
        #print(corr)
        #print(reg.coef_)
        #print(reg.intercept_)
    return derivs

def comp_other_methods_hists(ax,title,s_s_e_auc_df,fc_g_e_auc_df,viper_auc_df,scenic_auc_df):
    names = ['S-S','FC-G','viper','AUCell']
    lines = [s_s_e_auc_df,fc_g_e_auc_df,viper_auc_df,scenic_auc_df]
    derivs = quantify_method_increase(ax,lines,names)
    colors = ['orange','blue','red','black']
    derivs = pd.DataFrame(derivs)
    derivs = pd.melt(derivs,value_vars=derivs.columns,var_name='Method',value_name='Slope')
    sns.kdeplot(data=derivs,x='Slope',hue='Method',fill=True,palette=colors,legend=True,ax=ax)


def comp_other_methods(ax,title,s_s_e_auc_df,fc_g_e_auc_df,viper_auc_df,scenic_auc_df,ymin=0.45,ymax=1,legend=True):
    sns.lineplot(x='cutoff',y='AUC',data=s_s_e_auc_df,markers=True,color='orange',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=s_s_e_auc_df,markers=True,color='orange',ax=ax,edgecolor='w',linewidth=1.5,marker='o',label='S-S Ensemble')

    sns.lineplot(x='cutoff',y='AUC',data=fc_g_e_auc_df,markers=True,color='blue',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=fc_g_e_auc_df,markers=True,color='blue',ax=ax,marker='v',label='FC-G Ensemble')

    sns.lineplot(x='cutoff',y='AUC',data=viper_auc_df,markers=True,color='red',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=viper_auc_df,markers=True,color='red',ax=ax,edgecolor='w',linewidth=1.5,marker='x',label='viper')

    sns.lineplot(x='cutoff',y='AUC',data=scenic_auc_df,markers=True,color='black',ax=ax)
    sns.scatterplot(x='cutoff',y='AUC',data=scenic_auc_df,markers=True,color='black',ax=ax,marker='d',label='AUCell')

    ax.set_ylim(ymin,ymax)
    ax.set_ylabel('ROC AUC')
    ax.set_xlabel('Reconstruction Correlation\nRanking Cutoff')

    #ax.set_title(title)

    if legend:
        ax.legend().set_zorder(0)
    else:
        ax.legend().set_visible(False)

    ax.axhline(y=0.5,color='lightcoral',linestyle='--',zorder=0)
"""
fig,ax = plt.subplots()
fig.set_figwidth(7)
fig.set_figheight(5)
ko_tf_plot(ax,s_s_e_auc_df,fc_g_e_auc_df,viper_auc_df,scenic_auc_df,viper_s_s_df)
fig.savefig('knocktf_rel_vs_auc_frac.png')
"""

###DOROTHEA BENCHMARK DATA
def dorothea_plot():
    viper_activities = pd.read_csv('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/knn_inferred_tf_activities/VIPERdiff_activities.csv',sep='\t',index_col=0)
    scenic_activities = pd.read_csv('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/knn_inferred_tf_activities/SCENICdiff_activities.csv',sep='\t',index_col=0)
    s_s_activities = [pd.read_csv(f,index_col=0)for f in dorothea_s_s_activity_files]
    fc_g_activities = [pd.read_csv(f,index_col=0)for f in dorothea_fc_g_activity_files]
    s_s_ensemble = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/shallow_shallow/dorothea_ensemble_activities.csv',sep='\t',index_col=0) 
    fc_g_ensemble = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/fc_g/dorothea_ensemble_activities.csv',sep='\t',index_col=0)
    id_to_kotf = pd.read_pickle('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/knn_inferred_tf_activities/id_to_kotf.pkl')

    id_to_kotf = pd.DataFrame({i:i.split('_')[-1] for i in s_s_activities[0].index},index=['TF']).T

    viper_s_s = consensus([viper_activities,s_s_ensemble])
    viper_fc_g = consensus([viper_activities,fc_g_ensemble])
    fc_g_s_s_consensus = consensus([fc_g_ensemble,s_s_ensemble])
    scenic_s_s_consensus = consensus([scenic_activities,s_s_ensemble])
    scenic_fc_g_consensus = consensus([scenic_activities,fc_g_ensemble])

    s_s_values, s_s_names = get_recon_corr_of_sample(s_s_activities,id_to_kotf,dorothea_recon_corr_s_s)
    fc_g_values, fc_g_names = get_recon_corr_of_sample(fc_g_activities,id_to_kotf,dorothea_recon_corr_fc_g)

    #s_s_values, s_s_names = get_variance_of_sample_across_models(s_s_activities,id_to_kotf)
    #fc_g_values, fc_g_names = get_variance_of_sample_across_models(fc_g_activities,id_to_kotf)

    #s_s_values, s_s_names = tf_attr,tf_sum_names 
    #fc_g_values, fc_g_names = tf_attr,tf_sum_names

    #s_s_values, s_s_names = tfs_rel,tfs_in_pkn 
    #fc_g_values, fc_g_names = tfs_rel,tfs_in_pkn 

    s_s_aucs = []
    s_s_counts = []
    for df in s_s_activities:
        aucs,counts = get_ranking_cutoff_aucs(df,cutoffs,id_to_kotf,s_s_values,s_s_names)
        s_s_aucs.append(aucs)
        s_s_counts.append(counts)

    fc_g_aucs = []
    fc_g_counts = []
    for df in fc_g_activities:
        aucs,counts = get_ranking_cutoff_aucs(df,cutoffs,id_to_kotf,fc_g_values,fc_g_names)
        fc_g_aucs.append(aucs)
        fc_g_counts.append(counts)

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

    viper_aucs, viper_counts = get_ranking_cutoff_aucs(viper_activities,cutoffs,id_to_kotf,s_s_values,s_s_names)
    scenic_aucs, scenic_counts = get_ranking_cutoff_aucs(scenic_activities,cutoffs,id_to_kotf,s_s_values,s_s_names)
    s_s_e_aucs, s_s_e_counts = get_ranking_cutoff_aucs(s_s_ensemble,cutoffs,id_to_kotf,s_s_values,s_s_names)
    fc_g_e_aucs, fc_g_e_counts = get_ranking_cutoff_aucs(fc_g_ensemble,cutoffs,id_to_kotf,fc_g_values,fc_g_names)
    viper_s_s_aucs, viper_s_s_counts = get_ranking_cutoff_aucs(viper_s_s,cutoffs,id_to_kotf,s_s_values,s_s_names)
    viper_fc_g_aucs, viper_fc_g_counts = get_ranking_cutoff_aucs(viper_fc_g,cutoffs,id_to_kotf,fc_g_values,fc_g_names)
    fc_g_s_s_aucs, fc_g_s_s_counts = get_ranking_cutoff_aucs(fc_g_s_s_consensus,cutoffs,id_to_kotf,fc_g_values,fc_g_names)
    scenic_fc_g_aucs, scenic_fc_g_counts = get_ranking_cutoff_aucs(scenic_fc_g_consensus,cutoffs,id_to_kotf,fc_g_values,fc_g_names)
    scenic_s_s_aucs, scenic_s_s_counts = get_ranking_cutoff_aucs(scenic_s_s_consensus,cutoffs,id_to_kotf,s_s_values,s_s_names)

    viper_auc_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':viper_aucs,'Counts':viper_counts})
    scenic_auc_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':scenic_aucs,'Counts':scenic_counts})
    s_s_e_auc_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':s_s_e_aucs,'Counts':s_s_e_counts})
    fc_g_e_auc_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':fc_g_e_aucs,'Counts':fc_g_e_counts})
    viper_s_s_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':viper_s_s_aucs,'Counts':viper_s_s_counts})
    viper_fc_g_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':viper_fc_g_aucs,'Counts':viper_fc_g_counts})
    fc_g_s_s_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':fc_g_s_s_aucs,'Counts':fc_g_s_s_counts})
    scenic_fc_g_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':scenic_fc_g_aucs,'Counts':scenic_fc_g_counts})
    scenic_s_s_df = pd.DataFrame({'cutoff':[c for c in cutoffs],'AUC':scenic_s_s_aucs,'Counts':scenic_s_s_counts})

    return {'s_s_e_auc_df':s_s_e_auc_df,'fc_g_e_auc_df':fc_g_e_auc_df,'viper_auc_df':viper_auc_df,'scenic_auc_df': scenic_auc_df,'viper_s_s_df': viper_s_s_df,'viper_fc_g_df': viper_fc_g_df,'fc_g_s_s_df': fc_g_s_s_df,'scenic_fc_g_df':scenic_fc_g_df, 'scenic_s_s_df':scenic_s_s_df}


#fig,ax = plt.subplots(1,2,sharey=True)
#fig.set_figwidth(10)
#fig.set_figheight(5)

dorothea_res = dorothea_plot()
knocktf_res = knocktf_plot()
pd.to_pickle(dorothea_res,full_path+"dorothea_corr_res.pkl")
pd.to_pickle(knocktf_res,full_path+"knocktf_corr_res.pkl")

dorothea_res = pd.read_pickle(full_path+"dorothea_corr_res.pkl")
knocktf_res = pd.read_pickle(full_path+"knocktf_corr_res.pkl")

#comp_other_methods((ax,title,s_s_e_auc_df,fc_g_e_auc_df,viper_auc_df,scenic_auc_df,legend=True)
#s_s_consensus_plot(ax,title,s_s_e_auc_df,viper_auc_df,viper_s_s_df,legend=True)

def dorothea_comp_other_methods_plot(ax,ymin=0.45,ymax=1,legend=False):
    comp_other_methods(ax,'Perturbation Validation',dorothea_res['s_s_e_auc_df'],dorothea_res['fc_g_e_auc_df'],dorothea_res['viper_auc_df'],dorothea_res['scenic_auc_df'],ymin=ymin,ymax=ymax,legend=legend)

def dorothea_s_s_viper_consensus(ax,ymin=0.45,ymax=1,legend=False):
    s_s_consensus_plot(ax,'Perturbation Validation',dorothea_res['s_s_e_auc_df'],dorothea_res['viper_auc_df'],dorothea_res['viper_s_s_df'],ymin=ymin,ymax=ymax,legend=legend)

def dorothea_fc_g_viper_consensus(ax,ymin=0.45,ymax=1,legend=False):
    fc_g_consensus_plot(ax,'Perturbation Validation',dorothea_res['fc_g_e_auc_df'],dorothea_res['viper_auc_df'],dorothea_res['viper_fc_g_df'],ymin=ymin,ymax=ymax,legend=legend)

def dorothea_fc_g_s_s_consensus(ax,ymin=0.45,ymax=1,legend=False):
    fc_g_s_s_consensus_plot(ax,'Perturbation Validation',dorothea_res['fc_g_e_auc_df'],dorothea_res['s_s_e_auc_df'],dorothea_res['fc_g_s_s_df'],ymin=ymin,ymax=ymax,legend=legend)
            
def dorothea_s_s_scenic_consensus(ax,ymin=0.45,ymax=1,legend=False):
    s_s_scenic_consensus_plot(ax,'Perturbation Validation',dorothea_res['s_s_e_auc_df'],dorothea_res['scenic_auc_df'],dorothea_res['scenic_s_s_df'],ymin=ymin,ymax=ymax,legend=legend)

def dorothea_fc_g_scenic_consensus(ax,ymin=0.45,ymax=1,legend=False):
    fc_g_scenic_consensus_plot(ax,'Perturbation Validation', dorothea_res['fc_g_e_auc_df'],dorothea_res['scenic_auc_df'],dorothea_res['scenic_fc_g_df'],ymin=ymin,ymax=ymax,legend=legend)


def dorothea_comp_other_methods_hists(ax):
    comp_other_methods_hists(ax,'Perturbation Validation',dorothea_res['s_s_e_auc_df'],dorothea_res['fc_g_e_auc_df'],dorothea_res['viper_auc_df'],dorothea_res['scenic_auc_df'])

def dorothea_s_s_fc_g_hists(ax):
    fc_g_s_s_consensus_hists(ax,'Perturbation Validation',dorothea_res['fc_g_e_auc_df'],dorothea_res['s_s_e_auc_df'],dorothea_res['fc_g_s_s_df'])

def dorothea_s_s_viper_hists(ax):
    s_s_viper_consensus_hists(ax,'Perturbation Validation',dorothea_res['s_s_e_auc_df'],dorothea_res['viper_auc_df'],dorothea_res['viper_s_s_df'])

def dorothea_fc_g_viper_hists(ax):
    fc_g_viper_consensus_hists(ax,'Perturbation Validation',dorothea_res['fc_g_e_auc_df'],dorothea_res['viper_auc_df'],dorothea_res['viper_fc_g_df'])

def dorothea_s_s_scenic_hists(ax):
    s_s_scenic_consensus_hists(ax,'Perturbation Validation',dorothea_res['s_s_e_auc_df'],dorothea_res['scenic_auc_df'],dorothea_res['scenic_s_s_df'])

def dorothea_fc_g_scenic_hists(ax):
    fc_g_scenic_consensus_hists(ax,'Perturbation Validation', dorothea_res['fc_g_e_auc_df'],dorothea_res['scenic_auc_df'],dorothea_res['scenic_fc_g_df'])
    


def knocktf_comp_other_methods_plot(ax,ymin=0.45,ymax=1,legend=False):
    comp_other_methods(ax,'KnockTF Validation',knocktf_res['s_s_e_auc_df'],knocktf_res['fc_g_e_auc_df'],knocktf_res['viper_auc_df'],knocktf_res['scenic_auc_df'],ymin=ymin,ymax=ymax,legend=legend)

def knocktf_s_s_viper_consensus(ax,ymin=0.45,ymax=1,legend=False):
    s_s_consensus_plot(ax,'KnockTF Validation',knocktf_res['s_s_e_auc_df'],knocktf_res['viper_auc_df'],knocktf_res['viper_s_s_df'],ymin=ymin,ymax=ymax,legend=legend)

def knocktf_fc_g_viper_consensus(ax,ymin=0.45,ymax=1,legend=False):
    fc_g_consensus_plot(ax,'KnockTF Validation',knocktf_res['fc_g_e_auc_df'],knocktf_res['viper_auc_df'],knocktf_res['viper_fc_g_df'],ymin=ymin,ymax=ymax,legend=legend)

def knocktf_fc_g_s_s_consensus(ax,ymin=0.45,ymax=1,legend=False):
    fc_g_s_s_consensus_plot(ax,'KnockTF Validation',knocktf_res['fc_g_e_auc_df'],knocktf_res['s_s_e_auc_df'],knocktf_res['fc_g_s_s_df'],ymin=ymin,ymax=ymax,legend=legend)

def knocktf_s_s_scenic_consensus(ax,ymin=0.45,ymax=1,legend=False):
    s_s_scenic_consensus_plot(ax,'KnockTF Validation',knocktf_res['s_s_e_auc_df'],knocktf_res['scenic_auc_df'],knocktf_res['scenic_s_s_df'],ymin=ymin,ymax=ymax,legend=legend)

def knocktf_fc_g_scenic_consensus(ax,ymin=0.45,ymax=1,legend=False):
    fc_g_scenic_consensus_plot(ax,'KnockTF Validation',knocktf_res['fc_g_e_auc_df'],knocktf_res['scenic_auc_df'],knocktf_res['scenic_fc_g_df'],ymin=ymin,ymax=ymax,legend=legend)


def knocktf_comp_other_methods_hists(ax):
    comp_other_methods_hists(ax,'KnockTF Validation',knocktf_res['s_s_e_auc_df'],knocktf_res['fc_g_e_auc_df'],knocktf_res['viper_auc_df'],knocktf_res['scenic_auc_df'])

def knocktf_s_s_fc_g_hists(ax):
    fc_g_s_s_consensus_hists(ax,'KnockTF Validation',knocktf_res['fc_g_e_auc_df'],knocktf_res['s_s_e_auc_df'],knocktf_res['fc_g_s_s_df'])

def knocktf_s_s_viper_hists(ax):
    s_s_viper_consensus_hists(ax,'KnockTF Validation',knocktf_res['s_s_e_auc_df'],knocktf_res['viper_auc_df'],knocktf_res['viper_s_s_df'])

def knocktf_fc_g_viper_hists(ax):
    fc_g_viper_consensus_hists(ax,'KnockTF Validation',knocktf_res['fc_g_e_auc_df'],knocktf_res['viper_auc_df'],knocktf_res['viper_fc_g_df'])

def knocktf_s_s_scenic_hists(ax):
    s_s_scenic_consensus_hists(ax,'KnockTF Validation',knocktf_res['s_s_e_auc_df'],knocktf_res['scenic_auc_df'],knocktf_res['scenic_s_s_df'])

def knocktf_fc_g_scenic_hists(ax):
    fc_g_scenic_consensus_hists(ax,'KnockTF Validation', knocktf_res['fc_g_e_auc_df'],knocktf_res['scenic_auc_df'],knocktf_res['scenic_fc_g_df'])
#ko_tf_plot(ax[0],'Sample Reconstruction Correlation Ranking\nvs. DoRothEA Benchmark Performance',*dorothea_res,legend=False)
#ko_tf_plot(ax[1],'Sample Reconstruction Correlation Ranking\nvs. KnockTF Benchmark Performance',*knocktf_res)
#fig.savefig('corr_vs_auc_frac.png',dpi=300)
