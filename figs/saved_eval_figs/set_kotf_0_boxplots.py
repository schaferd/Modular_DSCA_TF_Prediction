import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from scipy import stats

sys.path.insert(1,'/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/')
from get_roc_curve import getROC
from get_pert_roc_curves import getPertROC

SWARM_PLOT_SCATTER_SIZE=4
SWARM_PLOT_ALPHA=0.7
SWARM_PLOT_COLOR='midnightblue'
BOX_PLOT_OUTLINE_COLOR='k'
BOX_PLOT_WIDTH=0.5
BOX_PLOT_ORDER=0
BOX_PLOT_LINEWIDTH=1
BOX_PROPS = {'facecolor': '0.9', 'edgecolor':BOX_PLOT_OUTLINE_COLOR,"linewidth":BOX_PLOT_LINEWIDTH}
MEDIAN_PROPS = {'color':BOX_PLOT_OUTLINE_COLOR,"zorder":3,"linewidth":BOX_PLOT_LINEWIDTH}
WHISKER_PROPS = {'color':BOX_PLOT_OUTLINE_COLOR,"linewidth":BOX_PLOT_LINEWIDTH}
CAP_PROPS = {'color':BOX_PLOT_OUTLINE_COLOR,"linewidth":BOX_PLOT_LINEWIDTH}

base_path= '/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/'
s_s_path = [base_path+'shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_14.26.53/',base_path+'/shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/']
fc_g_path = [base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.56/',base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/']

ko0_s_s_path = [base_path+'/set_kotf_to_0/shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_14.26.53/',base_path+'/set_kotf_to_0/shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/']

ko0_fc_g_path = [base_path+'/set_kotf_to_0/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.56/',base_path+'/set_kotf_to_0/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/']

s_s_activity_files = []
fc_g_activity_files = []
dorothea_s_s_activity_files = []
dorothea_fc_g_activity_files = []

ko0_s_s_activity_files = []
ko0_fc_g_activity_files = []
ko0_dorothea_s_s_activity_files = []
ko0_dorothea_fc_g_activity_files = []

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


for path in ko0_s_s_path:
    fold_paths = []
    fold_paths = fold_paths+[path+'/'+p for p in os.listdir(path) if 'fold' in p]
    act_path = []
    for p in fold_paths:
        act_path = act_path + [p+'/'+i for i in os.listdir(p) if 'ko_activities' in i]
        for p2 in act_path:
            ko0_s_s_activity_files = ko0_s_s_activity_files+[p2+'/'+f for f in os.listdir(p2) if 'knockout_diff_activities.csv' in f]
            ko0_dorothea_s_s_activity_files = ko0_dorothea_s_s_activity_files+[p2+'/'+f for f in os.listdir(p2) if ('diff_activities.csv' in f and 'knockout' not in f)]


for path in ko0_fc_g_path:
    fold_paths = []
    fold_paths = fold_paths+[path+'/'+p for p in os.listdir(path) if 'fold' in p]
    act_path = []
    for p in fold_paths:
        act_path = act_path + [p+'/'+i for i in os.listdir(p) if 'ko_activities' in i]
        for p2 in act_path:
            ko0_fc_g_activity_files = ko0_fc_g_activity_files+[p2+'/'+f for f in os.listdir(p2) if 'knockout_diff_activities.csv' in f]
            ko0_dorothea_fc_g_activity_files = ko0_dorothea_fc_g_activity_files+[p2+'/'+f for f in os.listdir(p2) if ('diff_activities.csv' in f and 'knockout' not in f)]

s_s_activity_files = list(set(s_s_activity_files))
fc_g_activity_files = list(set(fc_g_activity_files))
dorothea_s_s_activity_files = list(set(dorothea_s_s_activity_files))
dorothea_fc_g_activity_files = list(set(dorothea_fc_g_activity_files))

ko0_s_s_activity_files = list(set(ko0_s_s_activity_files))
ko0_fc_g_activity_files = list(set(ko0_fc_g_activity_files))
ko0_dorothea_s_s_activity_files = list(set(ko0_dorothea_s_s_activity_files))
ko0_dorothea_fc_g_activity_files = list(set(ko0_dorothea_fc_g_activity_files))

print("len s_s_activity files",len(s_s_activity_files))
print("len fc_g_activity files",len(fc_g_activity_files))

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
viper_fc_g_consensus = consensus([fc_g_ensemble,viper_activities])

def get_aucs(activities,id_to_kotf):
    aucs = getPertROC(activities,id_to_kotf,None).auc
    return aucs


def all_sample_box_plot(ax,title,dorothea_ae_df,viper_auc,scenic_auc,viper_s_s_auc,viper_fc_g_auc,show_legend=False,ymin=0.48,ymax=1):
    sns.boxplot(data=dorothea_ae_df,x='model',y='AUC',ax=ax,color='w',notch=False,showfliers=False,width=BOX_PLOT_WIDTH,boxprops=BOX_PROPS,medianprops=MEDIAN_PROPS,whiskerprops=WHISKER_PROPS,capprops=CAP_PROPS)
    sns.swarmplot(data=dorothea_ae_df,x='model',y='AUC',ax=ax,color=SWARM_PLOT_COLOR,size=SWARM_PLOT_SCATTER_SIZE,alpha=SWARM_PLOT_ALPHA)
    """
    ax.axhline(y=viper_auc,color='forestgreen',linestyle='--',zorder=0,label='viper')
    ax.axhline(y=scenic_auc,color='mediumblue',linestyle='dashdot',zorder=0,label='AUCell')
    ax.axhline(y=viper_s_s_auc,color='purple',linestyle='dotted',zorder=0,label='viper-S-S Consensus')
    """
    ax.axhline(y=viper_auc,color='0.6',linestyle='--',zorder=0,label='viper')
    ax.axhline(y=scenic_auc,color='0.4',linestyle='dashdot',zorder=0,label='AUCell')
    #ax.axhline(y=viper_s_s_auc,color='0.2',linestyle='dotted',zorder=0,label='viper-S-S Consensus')
    #ax.axhline(y=viper_fc_g_auc,color='0.5',linestyle=(0, (3, 1, 1, 1, 1, 1)),zorder=0,label='viper-FC-G Consensus')
    ax.axhline(y=0.5,color='lightcoral',linestyle='--',zorder=0)
    """
    y_axis_markers = [viper_auc,scenic_auc,viper_s_s_auc,viper_fc_g_auc]
    for point in y_axis_markers:
        y_val = point
        plt.annotate(f'$\\bigtriangleup$', (0,y_val),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center',
                 fontsize=12)
    """

    ax.set_ylim(ymin,ymax)
    ax.set_ylabel('ROC AUC')
    ax.set_xlabel('Model Type')
    #ax.set_title(title)
    if show_legend:
        ax.legend(loc='upper right')
    else:
        ax.legend().set_visible(False)



########################################################
###DOROTHEA BENCHMARK DATA
########################################################
def ktf():
    temp_id_to_kotf = id_to_kotf.loc[viper_s_s_consensus.index,'TF']
    ko_viper_aucs = get_aucs(viper_activities.loc[viper_s_s_consensus.index,viper_s_s_consensus.columns],temp_id_to_kotf)
    ko_scenic_aucs = get_aucs(scenic_activities.loc[viper_s_s_consensus.index,viper_s_s_consensus.columns],temp_id_to_kotf)
    ko_viper_s_s_aucs = get_aucs(viper_s_s_consensus,temp_id_to_kotf)
    ko_viper_fc_g_aucs = get_aucs(viper_fc_g_consensus,temp_id_to_kotf)
    ko_s_s_e_aucs = get_aucs(s_s_ensemble,temp_id_to_kotf)
    ko_s_s_aucs = [get_aucs(df.loc[viper_s_s_consensus.index,viper_s_s_consensus.columns],temp_id_to_kotf) for df in s_s_activities]
    ko_fc_g_aucs = [get_aucs(df.loc[viper_s_s_consensus.index,viper_s_s_consensus.columns],temp_id_to_kotf) for df in fc_g_activities]

    ko_ae_df = pd.DataFrame({'S-S':ko_s_s_aucs,'FC-G':ko_fc_g_aucs}).melt(value_vars=['S-S','FC-G'], var_name='model',value_name='AUC')

    ko_ae_df.to_pickle('ko_ae_df.pkl')
    ko_ae_df = pd.read_pickle('ko_ae_df.pkl')
    
    ###t-test
    pval = stats.ttest_ind(ko_s_s_aucs,ko_fc_g_aucs,alternative='greater')[1]*5
    print('knocktf S-S, FC-G, t-test pval '+str(pval))
    pval = stats.ttest_1samp(ko_s_s_aucs,ko_viper_aucs,alternative='less')[1]*5
    print('knocktf S-S, viper, t-test pval '+str(pval))
    pval = stats.ttest_1samp(ko_fc_g_aucs,ko_viper_aucs,alternative='less')[1]*5
    print('knocktf FC-G, viper, t-test pval '+str(pval))
    pval = stats.ttest_1samp(ko_s_s_aucs,ko_scenic_aucs,alternative='greater')[1]*5
    print('knocktf S-S, scenic, t-test pval '+str(pval))
    pval = stats.ttest_1samp(ko_fc_g_aucs,ko_scenic_aucs,alternative='greater')[1]*5
    print('knocktf FC-G, scenic, t-test pval '+str(pval))

    return ko_ae_df,ko_viper_aucs,ko_scenic_aucs,ko_viper_s_s_aucs,ko_viper_fc_g_aucs



########################################################
###DOROTHEA BENCHMARK DATA
########################################################
def dorothea():
    viper_activities = pd.read_csv('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/knn_inferred_tf_activities/VIPERdiff_activities.csv',sep='\t',index_col=0)
    scenic_activities = pd.read_csv('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/knn_inferred_tf_activities/SCENICdiff_activities.csv',sep='\t',index_col=0)
    dorothea_s_s_activities = [pd.read_csv(f,index_col=0)for f in dorothea_s_s_activity_files]
    dorothea_fc_g_activities = [pd.read_csv(f,index_col=0)for f in dorothea_fc_g_activity_files]
    dorothea_s_s_e = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/shallow_shallow/dorothea_ensemble_activities.csv',sep='\t',index_col=0) 
    dorothea_fc_g_e = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/fc_g/dorothea_ensemble_activities.csv',sep='\t',index_col=0)
    id_to_kotf = pd.read_pickle('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/knn_inferred_tf_activities/id_to_kotf.pkl')

    viper_s_s = consensus([viper_activities,dorothea_s_s_e])
    viper_fc_g = consensus([viper_activities,dorothea_fc_g_e])

    #temp_id_to_kotf = id_to_kotf.loc[viper_s_s.index,'TF']
    temp_id_to_kotf = id_to_kotf

    dorothea_viper_auc = get_aucs(viper_activities.loc[viper_s_s.index,viper_s_s.columns],temp_id_to_kotf)
    dorothea_scenic_auc = get_aucs(scenic_activities.loc[viper_s_s.index,viper_s_s.columns],temp_id_to_kotf)
    dorothea_viper_s_s_auc = get_aucs(viper_s_s,temp_id_to_kotf)
    dorothea_viper_fc_g_auc = get_aucs(viper_fc_g,temp_id_to_kotf)
    dorothea_s_s_e_auc = get_aucs(dorothea_s_s_e,temp_id_to_kotf)
    dorothea_s_s_aucs = [get_aucs(df.loc[viper_s_s.index,viper_s_s.columns],temp_id_to_kotf) for df in dorothea_s_s_activities]
    dorothea_fc_g_aucs = [get_aucs(df.loc[viper_s_s.index,viper_s_s.columns],temp_id_to_kotf) for df in dorothea_fc_g_activities]

    dorothea_ae_df = pd.DataFrame({'S-S':dorothea_s_s_aucs,'FC-G':dorothea_fc_g_aucs}).melt(value_vars=['S-S','FC-G'], var_name='model',value_name='AUC')

    dorothea_ae_df.to_pickle('dorothea_ae_df.pkl')
    dorothea_ae_df = pd.read_pickle('dorothea_ae_df.pkl')

    ###t-test
    pval = stats.ttest_ind(dorothea_s_s_aucs,dorothea_fc_g_aucs,alternative='greater')[1]*5
    print('S-S, FC-G, t-test pval '+str(pval))
    pval = stats.ttest_1samp(dorothea_s_s_aucs,dorothea_viper_auc,alternative='greater')[1]*5
    print('S-S, viper, t-test pval '+str(pval))
    pval = stats.ttest_1samp(dorothea_fc_g_aucs,dorothea_viper_auc,alternative='greater')[1]*5
    print('FC-G, viper, t-test pval '+str(pval))
    pval = stats.ttest_1samp(dorothea_s_s_aucs,dorothea_scenic_auc,alternative='greater')[1]*5
    print('S-S, scenic, t-test pval '+str(pval))
    pval = stats.ttest_1samp(dorothea_fc_g_aucs,dorothea_scenic_auc,alternative='greater')[1]*5
    print('FC-G, scenic, t-test pval '+str(pval))

    return dorothea_ae_df,dorothea_viper_auc,dorothea_scenic_auc,dorothea_viper_s_s_auc,dorothea_viper_fc_g_auc

########################################################
###DOROTHEA BENCHMARK DATA KOTF 0
########################################################


def dorothea_ko0():
    viper_activities = pd.read_csv('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/set_kotf_to_0_tf_activities/VIPERdiff_activities.csv',sep='\t',index_col=0)
    scenic_activities = pd.read_csv('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/set_kotf_to_0_tf_activities/SCENICdiff_activities.csv',sep='\t',index_col=0)
    dorothea_s_s_activities = [pd.read_csv(f,index_col=0)for f in ko0_dorothea_s_s_activity_files]
    dorothea_fc_g_activities = [pd.read_csv(f,index_col=0)for f in ko0_dorothea_fc_g_activity_files]
    dorothea_s_s_e = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/set_kotf_to_0/shallow_shallow/dorothea_ensemble_activities.csv',sep='\t',index_col=0) 
    dorothea_fc_g_e = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/set_kotf_to_0/fc_g/dorothea_ensemble_activities.csv',sep='\t',index_col=0)
    id_to_kotf = pd.read_pickle('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/set_kotf_to_0_tf_activities/id_to_kotf.pkl')

    viper_s_s = consensus([viper_activities,dorothea_s_s_e])
    viper_fc_g = consensus([viper_activities,dorothea_fc_g_e])

    #temp_id_to_kotf = id_to_kotf.loc[viper_s_s.index,'TF']
    temp_id_to_kotf = id_to_kotf

    ko0_dorothea_viper_auc = get_aucs(viper_activities.loc[viper_s_s.index,viper_s_s.columns],temp_id_to_kotf)
    ko0_dorothea_scenic_auc = get_aucs(scenic_activities.loc[viper_s_s.index,viper_s_s.columns],temp_id_to_kotf)
    ko0_dorothea_viper_s_s_auc = get_aucs(viper_s_s,temp_id_to_kotf)
    ko0_dorothea_viper_fc_g_auc = get_aucs(viper_fc_g,temp_id_to_kotf)
    ko0_dorothea_s_s_e_auc = get_aucs(dorothea_s_s_e,temp_id_to_kotf)
    ko0_dorothea_s_s_aucs = [get_aucs(df.loc[viper_s_s.index,viper_s_s.columns],temp_id_to_kotf) for df in dorothea_s_s_activities]
    ko0_dorothea_fc_g_aucs = [get_aucs(df.loc[viper_s_s.index,viper_s_s.columns],temp_id_to_kotf) for df in dorothea_fc_g_activities]

    ko0_dorothea_ae_df = pd.DataFrame({'S-S':ko0_dorothea_s_s_aucs,'FC-G':ko0_dorothea_fc_g_aucs}).melt(value_vars=['S-S','FC-G'], var_name='model',value_name='AUC')

    ko0_dorothea_ae_df.to_pickle('ko0_dorothea_ae_df.pkl')
    ko0_dorothea_ae_df = pd.read_pickle('ko0_dorothea_ae_df.pkl')

    return ko0_dorothea_ae_df,ko0_dorothea_viper_auc,ko0_dorothea_scenic_auc,ko0_dorothea_viper_s_s_auc,ko0_dorothea_viper_fc_g_auc


########################################################
### KNOCKTF DATA KOTF 0
########################################################

def ktf_ko0():

    viper_activities = pd.read_csv('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/viper_data/set_kotf_to_0_tf_activities/VIPERdiff_activities.csv',sep='\t',index_col=0)
    scenic_activities = pd.read_csv('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/viper_data/set_kotf_to_0_tf_activities/SCENICdiff_activities.csv',sep='\t',index_col=0)
    s_s_activities = [pd.read_csv(f,index_col=0)for f in ko0_s_s_activity_files]
    fc_g_activities = [pd.read_csv(f,index_col=0)for f in ko0_fc_g_activity_files]
    s_s_e = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/set_kotf_to_0/shallow_shallow/ensemble_activities.csv',sep='\t',index_col=0) 
    fc_g_e = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/set_kotf_to_0/fc_g/ensemble_activities.csv',sep='\t',index_col=0)
    d_id_to_kotf = pd.read_pickle('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/viper_data/set_kotf_to_0_tf_activities/id_to_kotf.pkl')

    viper_s_s = consensus([viper_activities,s_s_e])
    viper_fc_g = consensus([viper_activities,fc_g_e])

    temp_id_to_kotf = id_to_kotf.loc[viper_s_s.index,'TF']

    ko0_ktf_viper_auc = get_aucs(viper_activities.loc[viper_s_s.index,viper_s_s.columns],temp_id_to_kotf)
    ko0_ktf_scenic_auc = get_aucs(scenic_activities.loc[viper_s_s.index,viper_s_s.columns],temp_id_to_kotf)
    ko0_ktf_viper_s_s_auc = get_aucs(viper_s_s,temp_id_to_kotf)
    ko0_ktf_viper_fc_g_auc = get_aucs(viper_fc_g,temp_id_to_kotf)
    ko0_ktf_s_s_e_auc = get_aucs(s_s_e,temp_id_to_kotf)
    ko0_ktf_s_s_aucs = [get_aucs(df.loc[viper_s_s.index,viper_s_s.columns],temp_id_to_kotf) for df in s_s_activities]
    ko0_ktf_fc_g_aucs = [get_aucs(df.loc[viper_s_s.index,viper_s_s.columns],temp_id_to_kotf) for df in fc_g_activities]

    ko0_ktf_ae_df = pd.DataFrame({'S-S':ko0_ktf_s_s_aucs,'FC-G':ko0_ktf_fc_g_aucs}).melt(value_vars=['S-S','FC-G'], var_name='model',value_name='AUC')

    ko0_ktf_ae_df.to_pickle('ko0_ktf_ae_df.pkl')
    ko0_ktf_ae_df = pd.read_pickle('ko0_ktf_ae_df.pkl')

    return ko0_ktf_ae_df,ko0_ktf_viper_auc,ko0_ktf_scenic_auc,ko0_ktf_viper_s_s_auc,ko0_ktf_viper_fc_g_auc

def plot_dorothea_vanilla(ax,legend=False,ymin=0.48,ymax=1):
    dorothea_ = pd.read_pickle('dorothea_pert_val.pkl')
    all_sample_box_plot(ax,'Perturbation Validation',*dorothea_,show_legend=legend,ymin=ymin,ymax=ymax)

def plot_ktf_vanilla(ax,legend=False,ymin=0.48,ymax=1):
    ktf_ = pd.read_pickle('ktf_val.pkl')
    all_sample_box_plot(ax,'KnockTF Validation',*ktf_,show_legend=legend,ymin=ymin,ymax=ymax)

def plot_ko_results(ax):
    """
    res = dorothea()
    pd.to_pickle(res,'dorothea_pert_val.pkl')
    res = dorothea_ko0()
    pd.to_pickle(res,'dorothea_pert_val_ko0.pkl')
    res = ktf()
    pd.to_pickle(res,'ktf_val.pkl')
    res = ktf_ko0()
    pd.to_pickle(res,'ktf_val_ko0.pkl')
    """

    ktf_ = pd.read_pickle('ktf_val.pkl')
    ktf_ko0_ = pd.read_pickle('ktf_val_ko0.pkl')
    dorothea_ = pd.read_pickle('dorothea_pert_val.pkl')
    dorothea_ko0_ = pd.read_pickle('dorothea_pert_val_ko0.pkl')

    all_sample_box_plot(ax[0][0],'Perturbation Validation',*dorothea_,show_legend=True)
    all_sample_box_plot(ax[0][1],'Perturbation Validation\nPerturbed TF Set to 0',*dorothea_ko0_)
    all_sample_box_plot(ax[1][0],'Knock Out Validation',*ktf_)
    all_sample_box_plot(ax[1][1],'Knock Out Validation\nKO TF Set to 0',*ktf_ko0_)


#res = dorothea()
#pd.to_pickle(res,'dorothea_pert_val.pkl')
#res = ktf()
#raise ValueError()
#pd.to_pickle(res,'ktf_val.pkl')
"""
fig,ax = plt.subplots(2,2,sharey=True)
fig.subplots_adjust(left=0.1,bottom=0.2,right=0.9,top=0.8,wspace=0.35,hspace=0.5)
fig.set_figwidth(8)
fig.set_figheight(11)
plot_ko_results(ax)
#all_sample_box_plot(ax)
fig.savefig('all_boxplot.png',dpi=300)
"""
#dorothea()
#ktf()





