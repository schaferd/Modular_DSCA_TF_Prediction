import pandas as pd
import seaborn as sns
from scipy.stats import ttest_1samp, ttest_ind, kendalltau
import pickle as pkl
import os
import sys
from scipy import stats
from sklearn import metrics
import numpy as np
from data_processing import DataProcessing
from random import sample
import matplotlib.pyplot as plt

def calculate_consistency(activity_dir):

    raw_activities, index_to_ko_tfs = get_data_paths(activity_dir)
    print("CHECKING CONSISTENCY")

    rankings = [rank_filtered_ko_matrix(raw_activities[i],index_to_ko_tfs[i],scaled=True) for i in range(len(raw_activities))]
    rankings = [get_perturbation_info(rankings[i],index_to_ko_tfs[i]) for i in range(len(rankings))]
    rankings = [get_tfs_of_interest(i) for i in rankings]
    rankings = combine_samples(rankings)
    #distances = l2_dist(rankings)
    distances = kendall_w(rankings)

    #rand_distances = make_random_ranks(raw_activities,index_to_ko_tfs)
    #plot_distances_distribution(rand_distances.flatten(), distances)


    #stat, pval = ttest_ind(distances,rand_distances)
    #mean_diff = np.mean(rand_distances) - np.mean(distances)
    #print('mean diff', mean_diff)
    #info_dict = {'pval': pval, 'rand_distances':rand_distances, 'distances':distances, 'mean_diff':mean_diff}
    #with open(activity_dir+"/consistency_info.pkl", 'wb+') as f:
    #    pkl.dump(info_dict,f)

    return distances


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

    return raw_activities, index_to_ko_tfs

    
def rank_matrix(diff_activities,scaled=False):
    pert_tf_ids = list(diff_activities.index)
    ranked_matrix = diff_activities.rank(axis = 1,method='min',na_option='keep',ascending=True)
    ranked_matrix = ranked_matrix.reset_index(drop=True)
    if scaled == True:
        ranked_matrix = (ranked_matrix.T/ranked_matrix.max(axis=1)).T
    ranked_matrix.index = pert_tf_ids
    return ranked_matrix

def rank_filtered_ko_matrix(diff_activities,koed_tfs,scaled=False):
    koed_tfs_list = list(set(koed_tfs.values()))
    diff_activities = diff_activities.filter(items=koed_tfs_list)
    ranked_matrix = rank_matrix(diff_activities,scaled=scaled)
    return ranked_matrix

def make_random_ranks(run_dir,scaled=True,reps=20):
     
    dfs, koed_tfs = get_data_paths(run_dir)
    samples = None 
    for j in range(reps): 
        rand_rankings = [pd.DataFrame(np.random.rand(*df.to_numpy().shape), columns = df.columns, index=df.index) for df in dfs]
        rand_rankings = [rank_filtered_ko_matrix(rand_rankings[i],koed_tfs[i],scaled=scaled)for i in range(len(rand_rankings))]
        rand_rankings = [get_perturbation_info(rand_rankings[i],koed_tfs[i]) for i in range(len(rand_rankings))]
        rand_rankings = [get_tfs_of_interest(i) for i in rand_rankings]
        rand_rankings = combine_samples(rand_rankings)

        #dists = l2_dist(rand_rankings)
        dists = kendall_w(rand_rankings)
        if samples is None:
            samples = dists
        else:
            np.vstack((samples,dists))
    return samples

def plot_distances_distribution(null_distances, alt_distances):
    fig, ax = plt.subplots()

    ax.hist(null_distances,bins=20)
    ax.hist(alt_distances,bins=20)
    fig.savefig("consistency_dist_hist.png")

"""
Melt Rank df and add column with KO TF info
columns: rank, KO TF in sample, TF activity
"""
def get_perturbation_info(rankings,ko_tf_index):
    rankings['perturbed tf'] = [ko_tf_index[i] for i in rankings.index]
    value_vars = [col for col in rankings.columns if col != 'perturbed tf']
    rank_df = pd.melt(rankings,value_vars=value_vars,id_vars =['perturbed tf'],ignore_index=False)
    rank_df.rename({'value':'ranking'},axis=1,inplace=True)
    rank_df.rename({'variable':'regulon'},axis=1,inplace=True)

    return rank_df

"""
Only select Rankings of TFs where the TF has been knocked out
"""
def get_tfs_of_interest(perturbation_df):
    df_tf_of_interest = perturbation_df.copy()
    pert_tfs = set(df_tf_of_interest['perturbed tf'].tolist())
    pred_tfs = set(df_tf_of_interest['regulon'].tolist())
    tfs_of_interest = list(pert_tfs.intersection(pred_tfs))
    df_tf_of_interest = df_tf_of_interest[df_tf_of_interest['regulon'].isin(tfs_of_interest)]
    df_tf_of_interest['is tf perturbed'] = (df_tf_of_interest['regulon'] == df_tf_of_interest['perturbed tf'])
    koed_tfs_df = df_tf_of_interest.loc[df_tf_of_interest['is tf perturbed'] == True]
    koed_tfs_df = koed_tfs_df.drop(['regulon','perturbed tf', 'is tf perturbed'],axis=1).T

    return koed_tfs_df

def combine_samples(tfs_of_interest):
    df0 = pd.DataFrame(0,index=[0],columns = tfs_of_interest[0].columns)

    for df in tfs_of_interest:
        df0 = pd.concat([df0,df],ignore_index=True)

    new_df = df0.loc[1:,:]
    return new_df.to_numpy()

"""
Calculates pairwise l2 distances between rows of a matrix
Returns a symmetric matrix of distances between rows
"""
def l2_dist(rankings):
    distances = None
    for row in rankings:
        dist = np.apply_along_axis(np.linalg.norm, 1, rankings-row)
        if distances is None:
            distances = dist
        else:
            distances = np.vstack((distances,dist))
    distances = distances[np.triu_indices(distances.shape[0],k=1)]
    return distances

def kendall_w(rankings):
    kendall_ws = []
    for i,row_i in enumerate(rankings):
        for j,row_j in enumerate(rankings[i+1:]):
            if i != j:
                kendall_ws.append(kendalltau(row_i,row_j)[0])
    return kendall_ws


if __name__ == "__main__":
    #calculate_consistency('./',['/nobackup/users/schaferd/ae_project_outputs/model_eval/l2norm_test/__shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.0001_enl20.0001_moa1.0_rel_conn10_2-7_21.2.1/fold0_cycle0/ko_activities_cycle0_fold0/', '/nobackup/users/schaferd/ae_project_outputs/model_eval/l2norm_test_diff/__tffc-fc_epochs100_batchsize128_enlr0.001_delr0.0001_del20.0001_enl21e-05_moa1.0_rel_conn10_2-8_12.58.19/fold0_cycle0/ko_activities_cycle0_fold0'])
    calculate_consistency('/nobackup/users/schaferd/ae_project_outputs/model_eval/l2norm_test/__shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.0001_enl20.0001_moa1.0_rel_conn10_2-7_21.2.1/')
