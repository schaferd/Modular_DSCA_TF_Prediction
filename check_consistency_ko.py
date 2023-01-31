import pandas as pd
import seaborn as sns
import pickle as pkl
import os
import sys
from scipy import stats
from sklearn import metrics
import numpy as np
import torch 
from data_processing import DataProcessing
from random import sample
import matplotlib.pyplot as plt

is_gpu = False
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    is_gpu = True
print("is gpu "+str(is_gpu))


class Consistency():

    def __init__(self, save_path, activity_paths):
        self.savedir = save_path 
        self.raw_activities = []
        self.index_to_ko_tfs = []

        for i in activity_paths:
            self.raw_activities.append(self.load_diff_activities(i+'/diff_activities.csv'))
            with open(i+'/ko_tf_index.pkl','rb') as f:
                self.index_to_ko_tfs.append(pkl.load(f))

        print("CHECKING CONSISTENCY")
        self.scaled_rankings = [self.rank_matrix_only_pertTFs(self.raw_activities[i],self.index_to_ko_tfs[i]) for i in range(len(self.raw_activities))]
        self.perturbation_info = [self.get_perturbation_info(self.scaled_rankings[i],self.index_to_ko_tfs[i]) for i in range(len(self.scaled_rankings))]
        self.tfs_of_interest = [self.get_tfs_of_interest(i) for i in self.perturbation_info]
        self.results = self.combine_samples()

        self.mean = self.encoder_dist()
        
    def load_diff_activities(self,activity_file):
        df = pd.read_csv(activity_file,index_col=0)
        return df

    def rank_matrix(self, diff_activities):
        pert_tfs = list(diff_activities.index)
        ranked_matrix = diff_activities.rank(axis = 1,method='min',na_option='keep',ascending=True)
        ranked_matrix = ranked_matrix.reset_index(drop=True)
        scaled_rank_matrix = (ranked_matrix.T/ranked_matrix.max(axis=1)).T
        scaled_rank_matrix.index = pert_tfs

    def rank_matrix_only_pertTFs(self, diff_activities,koed_tfs):
        koed_tfs_list = list(set(koed_tfs.values()))
        pert_tfs = list(diff_activities.index)
        diff_activities = diff_activities.filter(items=koed_tfs_list)
        ranked_matrix = diff_activities.rank(axis = 1,method='min',na_option='keep',ascending=True)
        ranked_matrix = ranked_matrix.reset_index(drop=True)
        scaled_rank_matrix = (ranked_matrix.T/ranked_matrix.max(axis=1)).T
        scaled_rank_matrix.index = pert_tfs
        return scaled_rank_matrix

    def make_random_ranks(self):
        rand = np.random.rand(*self.results.shape)
        ranked = rand.argsort()
        ranked = ranked/np.amax(ranked,axis=1)[:,None]
        mean = np.mean(ranked,axis=0)
        std = np.std(ranked,axis=0)
        return mean,std

    def get_perturbation_info(self,scaled_rankings,ko_tf_index):
        rank_df = pd.melt(scaled_rankings,value_vars=scaled_rankings.columns,ignore_index=False)
        rank_df['perturbed tf'] = [ko_tf_index[i] for i in rank_df.index]
        rank_df.rename({'value':'scaled ranking'},axis=1,inplace=True)
        rank_df.rename({'variable':'regulon'},axis=1,inplace=True)

        return rank_df#, unscaled_rank_df

    def get_tfs_of_interest(self, perturbation_df):
        df_tf_of_interest = perturbation_df.copy()
        pert_tfs = set(df_tf_of_interest['perturbed tf'].tolist())
        pred_tfs = set(df_tf_of_interest['regulon'].tolist())
        tfs_of_interest = list(pert_tfs.intersection(pred_tfs))
        df_tf_of_interest = df_tf_of_interest[df_tf_of_interest['regulon'].isin(tfs_of_interest)]
        df_tf_of_interest['is tf perturbed'] = (df_tf_of_interest['regulon'] == df_tf_of_interest['perturbed tf'])
        koed_tfs_df = df_tf_of_interest.loc[df_tf_of_interest['is tf perturbed'] == True]
        koed_tfs_df = koed_tfs_df.drop(['regulon','perturbed tf', 'is tf perturbed'],axis=1).T

        return koed_tfs_df

    def combine_samples(self):
        df0 = pd.DataFrame(0,index=[0],columns = self.tfs_of_interest[0].columns)

        for df in self.tfs_of_interest:
            df0 = pd.concat([df0,df],ignore_index=True)

        new_df = df0.loc[1:,:]
        print(new_df)
        return new_df


    def encoder_dist(self,make_plot=True):
        mean = np.array(np.mean(self.results,axis=0))
        std = np.array(np.std(self.results,axis=0))

        rand_mean,rand_std = self.make_random_ranks()
        rand_order = rand_std.argsort()
        rand_mean = rand_mean[rand_order]
        rand_std = rand_std[rand_order]

        x = np.arange(1,mean.shape[0]+1)
        order = std.argsort()
        cv_mean = mean[order]
        cv_std = std[order]

        with open(self.savedir+"/consistency_std.pkl", 'wb+') as f:
            pkl.dump(cv_std,f)
        with open(self.savedir+"/consistency_rand_std.pkl", 'wb+') as f:
            pkl.dump(rand_std,f)

        auc_rand = metrics.auc(x,rand_std)
        auc = metrics.auc(x,cv_std)
        
        auc_diff = 0
        if auc < auc_rand:
            auc_diff = (auc_rand-auc)/auc_rand
        print("auc diff consistency: ",str(auc_diff))
        if make_plot == True:
            plt.clf()
            fig = plt.figure()
            fig.set_size_inches(4,4)
            plt.plot(x,cv_std,label="std")
            plt.plot(x,rand_std,label="random std")
            plt.title("consistency auc_diff: "+str(auc_diff))
            plt.legend()
            plt.savefig(self.savedir+'/consistency_plot.png')
        return mean

