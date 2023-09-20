import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.colors as colors

info_df = pd.read_csv('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/info.csv',sep='\t',index_col=0).set_index('sample_id')


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

def knocktf_data():

    viper_activity_df = pd.read_csv('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/viper_data/TF_activities/VIPERdiff_activities.csv',sep='\t',index_col=0)
    scenic_activity_df = pd.read_csv('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/viper_data/TF_activities/SCENICdiff_activities.csv',sep='\t',index_col=0)
    id_to_kotf = pd.read_pickle('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/fold0_cycle0/ko_activities_cycle0_fold0/knocktf_sample_to_tf.pkl')
    id_to_kotf = id_to_kotf.set_index('Sample_ID')
    s_s_ensemble = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/shallow_shallow/ensemble_activities.csv',sep='\t',index_col=0)
    fc_g_ensemble = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/fc_g/ensemble_activities.csv',sep='\t',index_col=0)

    viper_s_s_consensus = consensus([s_s_ensemble,viper_activity_df])

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

    knocktf_recon_corr_s_s = []
    knocktf_recon_corr_fc_g = []
    base_path= '/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/'
    s_s_path = [base_path+'shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_14.26.53/',base_path+'/shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/']
    fc_g_path = [base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.56/',base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/']

    for path in s_s_path:
        fold_paths = []
        fold_paths = fold_paths+[path+'/'+p for p in os.listdir(path) if 'fold' in p]
        act_path = []
        for p in fold_paths:
            act_path = act_path + [p+'/'+i for i in os.listdir(p) if 'ko_activities' in i]
            for p2 in act_path:
                knocktf_recon_corr_s_s = knocktf_recon_corr_s_s+[p2+'/'+f for f in os.listdir(p2) if 'knocktf_recon_corr' in f]


    for path in fc_g_path:
        fold_paths = []
        fold_paths = fold_paths+[path+'/'+p for p in os.listdir(path) if 'fold' in p]
        act_path = []
        for p in fold_paths:
            act_path = act_path + [p+'/'+i for i in os.listdir(p) if 'ko_activities' in i]
            for p2 in act_path:
                knocktf_recon_corr_fc_g = knocktf_recon_corr_fc_g+[p2+'/'+f for f in os.listdir(p2) if 'knocktf_recon_corr' in f]


    knocktf_recon_corr_s_s = [pd.read_pickle(f) for f in knocktf_recon_corr_s_s]
    knocktf_recon_corr_fc_g = [pd.read_pickle(f) for f in knocktf_recon_corr_fc_g]

    recon_corr_fc_g = pd.concat([pd.DataFrame(d,index=[0]) for d in knocktf_recon_corr_fc_g])
    recon_corr_fc_g = recon_corr_fc_g.mean()
    recon_corr_s_s = pd.concat([pd.DataFrame(d,index=[0]) for d in knocktf_recon_corr_s_s])
    recon_corr_s_s = recon_corr_s_s.mean()
    ko_tf_df['FC_G Recon Corr'] = recon_corr_fc_g
    ko_tf_df['S_S Recon Corr'] = recon_corr_s_s
    
    return ko_tf_df

def dorothea_data():

    viper_activity_df = pd.read_csv('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/knn_inferred_tf_activities/VIPERdiff_activities.csv',sep='\t',index_col=0)
    scenic_activity_df = pd.read_csv('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/knn_inferred_tf_activities/SCENICdiff_activities.csv',sep='\t',index_col=0)
    id_to_kotf = pd.read_pickle('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/fold0_cycle0/ko_activities_cycle0_fold0/pert_index_to_kotf.pkl')
    s_s_ensemble = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/shallow_shallow/dorothea_ensemble_activities.csv',sep='\t',index_col=0)
    fc_g_ensemble = pd.read_csv('/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/fc_g/dorothea_ensemble_activities.csv',sep='\t',index_col=0)

    viper_s_s_consensus = consensus([s_s_ensemble,viper_activity_df])

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
        tf = id_to_kotf[i]

        viper_ranking = viper_rank_df[tf].loc[i]
        s_s_ranking = s_s_rank_df[tf].loc[i]
        fc_g_ranking = fc_g_rank_df[tf].loc[i]
        scenic_ranking = scenic_rank_df[tf].loc[i]
        v_s_s_ranking = v_s_s_rank_df[tf].loc[i]

        ko_tf_rank_dict[i] = [s_s_ranking,fc_g_ranking,viper_ranking,scenic_ranking,v_s_s_ranking]

    ko_tf_df = pd.DataFrame(ko_tf_rank_dict).T.rename(columns={0:'S-S',1:'FC-G',2:'viper',3:'AUCell',4:'viper-S-S Consensus'})

    dorothea_recon_corr_s_s = []
    dorothea_recon_corr_fc_g = []
    base_path= '/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/'
    s_s_path = [base_path+'shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_14.26.53/',base_path+'/shallow_shallow/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/']
    fc_g_path = [base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.56/',base_path+'/fc_g/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/']

    for path in s_s_path:
        fold_paths = []
        fold_paths = fold_paths+[path+'/'+p for p in os.listdir(path) if 'fold' in p]
        act_path = []
        for p in fold_paths:
            act_path = act_path + [p+'/'+i for i in os.listdir(p) if 'ko_activities' in i]
            for p2 in act_path:
                dorothea_recon_corr_s_s = dorothea_recon_corr_s_s+[p2+'/'+f for f in os.listdir(p2) if 'dorothea_recon_corr' in f]


    for path in fc_g_path:
        fold_paths = []
        fold_paths = fold_paths+[path+'/'+p for p in os.listdir(path) if 'fold' in p]
        act_path = []
        for p in fold_paths:
            act_path = act_path + [p+'/'+i for i in os.listdir(p) if 'ko_activities' in i]
            for p2 in act_path:
                dorothea_recon_corr_fc_g = dorothea_recon_corr_fc_g+[p2+'/'+f for f in os.listdir(p2) if 'dorothea_recon_corr' in f]


    dorothea_recon_corr_s_s = [pd.read_pickle(f) for f in dorothea_recon_corr_s_s]
    dorothea_recon_corr_fc_g = [pd.read_pickle(f) for f in dorothea_recon_corr_fc_g]

    recon_corr_fc_g = pd.concat([pd.DataFrame(d,index=[0]) for d in dorothea_recon_corr_fc_g])
    recon_corr_fc_g = recon_corr_fc_g.mean()
    recon_corr_s_s = pd.concat([pd.DataFrame(d,index=[0]) for d in dorothea_recon_corr_s_s])
    recon_corr_s_s = recon_corr_s_s.mean()
    ko_tf_df['FC_G Recon Corr'] = recon_corr_fc_g
    ko_tf_df['S_S Recon Corr'] = recon_corr_s_s

    return ko_tf_df

knocktf_ko_tf_df = knocktf_data()
dorothea_ko_tf_df = dorothea_data()

LINE_COLOR = '0.6'
LINE_STYLE = '--'
MARKER_STYLE= 'o'
VMIN = 0
VMAX=0.4
COLORP = 'flare'
cmap = sns.cubehelix_palette(as_cmap=True)
CBAR_TEXT_ROTATION=270
CBAR_PAD=11
YMIN=-0.02
YMAX=1.02

def dorothea_s_s_viper_scatter(ax,legend=False,cax=None):
    x_axis = 'S-S'
    y_axis = 'viper'
    hue = 'S_S Recon Corr'
    data = dorothea_ko_tf_df.loc[:,[x_axis,y_axis,hue]]
    """
    data['density'] = 0
    data.loc[data[(data[x_axis] <= 0.5) & (data[y_axis] <= 0.5)].index,['density']] = len(data[(data[x_axis] <0.5) & (data[y_axis] < 0.5)])/len(data.index)
    data.loc[data[(data[x_axis] >= 0.5) & (data[y_axis] <= 0.5)].index,['density']] = len(data[(data[x_axis] >0.5) & (data[y_axis] < 0.5)])/len(data.index)
    data.loc[data[(data[x_axis] <= 0.5) & (data[y_axis] >= 0.5)].index,['density']] = len(data[(data[x_axis] <0.5) & (data[y_axis] > 0.5)])/len(data.index)
    data.loc[data[(data[x_axis] >= 0.5) & (data[y_axis] >= 0.5)].index,['density']] = len(data[(data[x_axis] >= 0.5) & (data[y_axis] >= 0.5)])/len(data.index)
    """

    scatter = ax.scatter(data[x_axis],data[y_axis],c=data[hue],cmap=cmap,marker=MARKER_STYLE,norm=colors.LogNorm(vmin=0.01,vmax=1))
    if legend:
        colorbar = plt.colorbar(scatter,label='Reconstruction Correlation',cax=cax)
    ax.axhline(y=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.axvline(x=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.set_xlim(YMIN,YMAX)
    ax.set_ylim(YMIN,YMAX)
    #ax.set_title('Perturbation Validation')
    ax.set_xlabel(x_axis+' ranking')
    ax.set_ylabel(y_axis+' ranking')

def dorothea_fc_g_s_s_scatter(ax,legend=False,cax=None):
    x_axis='FC-G'
    y_axis='S-S'
    data = dorothea_ko_tf_df.loc[:,[x_axis,y_axis]]
    scatter = ax.scatter(data[x_axis],data[y_axis],marker=MARKER_STYLE)
    ax.axhline(y=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.axvline(x=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.set_xlim(YMIN,YMAX)
    ax.set_ylim(YMIN,YMAX)
    ax.set_xlabel(x_axis+' ranking')
    ax.set_ylabel(y_axis+' ranking')

def dorothea_fc_g_viper_scatter(ax,legend=False,cax=None):
    x_axis = 'FC-G'
    y_axis = 'viper'
    hue = 'FC_G Recon Corr'
    data = dorothea_ko_tf_df.loc[:,[x_axis,y_axis,hue]]
    scatter = ax.scatter(data[x_axis],data[y_axis],c=data[hue],cmap=cmap, marker=MARKER_STYLE,norm=colors.SymLogNorm(0.01,vmin=0,vmax=1))
    if legend:
        colorbar = plt.colorbar(scatter,cax=cax)
        colorbar.set_label('Reconstruction Correlation',rotation=CBAR_TEXT_ROTATION,labelpad=CBAR_PAD)
    ax.axhline(y=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.axvline(x=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    #ax.scatter(knocktf_ko_tf_df['viper'],knocktf_ko_tf_df['FC-G'],marker='.')
    #ax.set_title('Perturbation Validation')
    ax.set_xlim(YMIN,YMAX)
    ax.set_ylim(YMIN,YMAX)
    ax.set_xlabel(x_axis+' ranking')
    ax.set_ylabel(y_axis+' ranking')

def dorothea_s_s_scenic_scatter(ax,legend=False,cax=None):
    x_axis='S-S'
    y_axis='AUCell'
    hue= 'S_S Recon Corr'
    data = dorothea_ko_tf_df.loc[:,[x_axis,y_axis,hue]]
    scatter = ax.scatter(data[x_axis],data[y_axis],c=data[hue],cmap=cmap, marker=MARKER_STYLE,norm=colors.SymLogNorm(0.01,vmin=0,vmax=1))
    if legend:
        colorbar = plt.colorbar(scatter,cax=cax)
        colorbar.set_label('Reconstruction Correlation',rotation=CBAR_TEXT_ROTATION,labelpad=CBAR_PAD)
    ax.axhline(y=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.axvline(x=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.set_xlim(YMIN,YMAX)
    ax.set_ylim(YMIN,YMAX)
    ax.set_xlabel(x_axis+' ranking')
    ax.set_ylabel(y_axis+' ranking')

def dorothea_fc_g_scenic_scatter(ax,legend=False,cax=None):
    x_axis='S-S'
    y_axis='AUCell'
    hue= 'S_S Recon Corr'
    data = dorothea_ko_tf_df.loc[:,[x_axis,y_axis,hue]]
    scatter = ax.scatter(data[x_axis],data[y_axis],c=data[hue],cmap=cmap, marker=MARKER_STYLE,norm=colors.SymLogNorm(0.01,vmin=0,vmax=1))
    if legend:
        colorbar = plt.colorbar(scatter,cax=cax)
        colorbar.set_label('Reconstruction Correlation',rotation=CBAR_TEXT_ROTATION,labelpad=CBAR_PAD)
    ax.axhline(y=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.axvline(x=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.set_xlim(YMIN,YMAX)
    ax.set_ylim(YMIN,YMAX)
    ax.set_xlabel(x_axis+' ranking')
    ax.set_ylabel(y_axis+' ranking')

def knocktf_fc_g_s_s_scatter(ax,legend=False,cax=None):
    x_axis='FC-G'
    y_axis='S-S'
    data = knocktf_ko_tf_df.loc[:,[x_axis,y_axis]]
    scatter = ax.scatter(data[x_axis],data[y_axis], marker=MARKER_STYLE)
    if legend:
        colorbar = plt.colorbar(scatter,cax=cax)
        colorbar.set_label('Reconstruction Correlation',rotation=CBAR_TEXT_ROTATION,labelpad=CBAR_PAD)
    ax.axhline(y=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.axvline(x=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.set_xlim(YMIN,YMAX)
    ax.set_ylim(YMIN,YMAX)
    #ax.set_title('KnockTF Validation')
    ax.set_xlabel(x_axis+' ranking')
    ax.set_ylabel(y_axis+' ranking')


def knocktf_s_s_viper_scatter(ax,legend=False,cax=None):
    x_axis = 'S-S'
    y_axis = 'viper'
    hue = 'S_S Recon Corr'
    data = knocktf_ko_tf_df.loc[:,[x_axis,y_axis,hue]]
    scatter = ax.scatter(data[x_axis],data[y_axis],c=data[hue],cmap=cmap, marker=MARKER_STYLE,norm=colors.SymLogNorm(0.01,vmin=0,vmax=1))
    if legend:
        colorbar = plt.colorbar(scatter,cax=cax)
        colorbar.set_label('Reconstruction Correlation',rotation=CBAR_TEXT_ROTATION,labelpad=CBAR_PAD)
    ax.axhline(y=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.axvline(x=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.set_xlim(YMIN,YMAX)
    ax.set_ylim(YMIN,YMAX)
    #ax.set_title('KnockTF Validation')
    ax.set_xlabel(x_axis+' ranking')
    ax.set_ylabel(y_axis+' ranking')

def knocktf_fc_g_viper_scatter(ax,legend=False,cax=None):
    x_axis = 'FC-G'
    y_axis = 'viper'
    hue = 'FC_G Recon Corr'
    data = knocktf_ko_tf_df.loc[:,[x_axis,y_axis,hue]]
    scatter = ax.scatter(data[x_axis],data[y_axis],c=data[hue],cmap=cmap, marker=MARKER_STYLE,norm=colors.SymLogNorm(0.01,vmin=0,vmax=1))
    if legend:
        colorbar = plt.colorbar(scatter,cax=cax)
        colorbar.set_label('Reconstruction Correlation',rotation=CBAR_TEXT_ROTATION,labelpad=CBAR_PAD)
    ax.axhline(y=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.axvline(x=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.set_xlim(YMIN,YMAX)
    ax.set_ylim(YMIN,YMAX)
    #ax.scatter(knocktf_ko_tf_df['viper'],knocktf_ko_tf_df['FC-G'],marker='.')
    #ax.set_title('KnockTF Validation')
    ax.set_xlabel(x_axis+' ranking')
    ax.set_ylabel(y_axis+' ranking')

def knocktf_s_s_scenic_scatter(ax,legend=False,cax=None):
    x_axis = 'S-S'
    y_axis = 'AUCell'
    hue = 'S_S Recon Corr'
    data = knocktf_ko_tf_df.loc[:,[x_axis,y_axis,hue]]
    scatter = ax.scatter(data[x_axis],data[y_axis],c=data[hue],cmap=cmap, marker=MARKER_STYLE,norm=colors.SymLogNorm(0.01,vmin=0,vmax=1))
    if legend:
        colorbar = plt.colorbar(scatter,cax=cax)
        colorbar.set_label('Reconstruction Correlation',rotation=CBAR_TEXT_ROTATION,labelpad=CBAR_PAD)
    ax.axhline(y=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.axvline(x=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.set_xlim(YMIN,YMAX)
    ax.set_ylim(YMIN,YMAX)
    #ax.scatter(knocktf_ko_tf_df['viper'],knocktf_ko_tf_df['FC-G'],marker='.')
    #ax.set_title('KnockTF Validation')
    ax.set_xlabel(x_axis+' ranking')
    ax.set_ylabel(y_axis+' ranking')

def knocktf_fc_g_scenic_scatter(ax,legend=False,cax=None):
    x_axis = 'FC-G'
    y_axis = 'AUCell'
    hue = 'FC_G Recon Corr'
    data = knocktf_ko_tf_df.loc[:,[x_axis,y_axis,hue]]
    scatter = ax.scatter(data[x_axis],data[y_axis],c=data[hue],cmap=cmap, marker=MARKER_STYLE,norm=colors.SymLogNorm(0.01,vmin=0,vmax=1))
    if legend:
        colorbar = plt.colorbar(scatter,cax=cax)
        colorbar.set_label('Reconstruction Correlation',rotation=CBAR_TEXT_ROTATION,labelpad=CBAR_PAD)
    ax.axhline(y=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.axvline(x=0.5,color=LINE_COLOR,linestyle=LINE_STYLE)
    ax.set_xlim(YMIN,YMAX)
    ax.set_ylim(YMIN,YMAX)
    #ax.set_title('KnockTF Validation')
    ax.set_xlabel(x_axis+' ranking')
    ax.set_ylabel(y_axis+' ranking')
