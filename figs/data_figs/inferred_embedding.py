import pandas as pd
from matplotlib.colors import LogNorm, Normalize
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import sys
import os
import numpy as np


base_path = '/nobackup/users/schaferd/ae_project_outputs/final_eval/'
shallow1 = base_path+'eval_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_2-22_9.5.16/'
shallow2 = base_path+'eval_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_2-22_9.5.15/'
deep1 = base_path+'eval_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_2-22_9.7.34/'
deep2 = base_path+'eval_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_2-22_9.7.57/'

#d_activites = np.mean(np.vstack([np.array(get_data_paths(deep1)[0]),np.array(get_data_paths(deep2)[0])]), axis=0)
#s_activities = np.mean(np.vstack([np.array(get_data_paths(shallow1)[0]),np.array(get_data_paths(shallow2)[0])]), axis=0)




def get_inferred_embedding(diff_activities):
    corr_coef = np.corrcoef(diff_activities,rowvar=False)
    return corr_coef

def get_data_paths(activity_dir):
    raw_activities = []
    index_to_ko_tfs = []

    activity_paths = [activity_dir+'/'+f for f in os.listdir(activity_dir) if "fold" in f and "cycle" in f and os.path.isfile(activity_dir+f) == False]
    ko_activity_paths = []
    for i, f in enumerate(activity_paths):
        paths = os.listdir(f)
        for p in paths:
            if "ko_activities" in p:
                ko_activity_paths.append(f+'/'+p+'/')
        

    for i in ko_activity_paths:
        raw_activities.append(pd.read_csv(i+'/diff_activities.csv',index_col=0))
        with open(i+'/ko_tf_index.pkl','rb') as f:
            index_to_ko_tfs.append(pkl.load(f))

    tfs = raw_activities[0].columns
    return raw_activities, tfs

def get_avg_corrcoef(activities,model_type):
    corr_coefs = None
    for i,a in enumerate(activities):
        corrs = get_inferred_embedding(a)
        #g = sns.clustermap(corrs,cmap='RdBu_r', vmin=-1, vmax=1)
        #g.fig.suptitle(model_type,fontsize='x-large')
        #g.fig.savefig(model_type+'_'+str(i)+'_clustermap.png')
        if corr_coefs is None:
            corr_coefs = np.array([corrs])
        else:
            corr_coefs = np.vstack([corr_coefs,[corrs]])
    corr_coefs = np.mean(corr_coefs,axis=0)
    return corr_coefs


s_tfs = get_data_paths(shallow1)[1]
#d_tfs = get_data_paths(deep1)[1]
#d_activities = np.vstack([np.array(get_data_paths(deep1)[0]),np.array(get_data_paths(deep2)[0])])
#s_activities = np.vstack([np.array(get_data_paths(shallow1)[0]),np.array(get_data_paths(shallow2)[0])])
s_activities = get_data_paths(shallow1)[0][0]
print(s_activities)
#d_corrs = pd.DataFrame(get_avg_corrcoef(d_activities,'deep'),index=d_tfs,columns=d_tfs)
#s_corrs = pd.DataFrame(get_avg_corrcoef(s_activities,'shallow'),index=s_tfs,columns=s_tfs)

#g1 = sns.clustermap(d_corrs,cmap='RdBu_r', vmin=-1, vmax=1)
#g1.fig.subplots_adjust(right=0.8,top=0.8)
#g1.ax_cbar.set_position((0.9,0.2,0.03,0.4))
#g1.fig.savefig('deep_clustermap.png',dpi=300)

plt.clf()
#g2 = sns.clustermap(s_corrs,cmap='RdBu_r', vmin=-1, vmax=1)
g2 = sns.clustermap(s_activities,cmap='RdBu_r',vmin=-2,vmax=2)

g2.fig.subplots_adjust(right=0.8,top=0.8)
g2.ax_cbar.set_position((0.9,0.2,0.03,0.4))
#g2.fig.savefig('shallow_clustermap.png',dpi=300)
g2.fig.savefig('activities_vs_samples_clustermap.png',dpi=300)

