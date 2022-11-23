import pandas as pd
import seaborn as sns
import pickle as pkl
import os
import sys
from scipy import stats
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

        self.raw_activities = [self.load_diff_activities(i+'/diff_activities.csv') for i in activity_paths]
        self.scaled_rankings = [self.rank_matrix(i) for i in self.raw_activities]
        self.perturbation_info = [self.get_perturbation_info(i) for i in self.scaled_rankings]
        self.tfs_of_interest = [self.get_tfs_of_interest(i) for i in self.perturbation_info]

        self.results = None
        self.CV = None
        
    def load_diff_activities(self,activity_file):
        #returns pandas df
        df = pd.read_csv(activity_file,index_col=0)
        #pert_tfs = list(df.index)
        return df

    def rank_matrix(self, diff_activities):
        pert_tfs = list(diff_activities.index)
        ranked_matrix = diff_activities.rank(axis = 1,method='min',na_option='keep',ascending=True)
        print('tfs', len(diff_activities.columns))
        ranked_matrix = ranked_matrix.reset_index(drop=True)
        scaled_rank_matrix = (ranked_matrix.T/ranked_matrix.max(axis=1)).T
        scaled_rank_matrix.index = pert_tfs
        print('scaled rank matrix')
        print(scaled_rank_matrix)
        return scaled_rank_matrix

    def make_random_cv(self):
        rand = np.random.rand(*self.results.shape)
        ranked = rand.argsort()
        mean = np.mean(ranked,axis=0)
        std = np.std(ranked,axis=0)
        rand_CV=std/mean
        with open(self.savedir+'/rand_CV_consistency.pkl','wb+') as f:
            pkl.dump(rand_CV, f)

    def get_perturbation_info(self,scaled_rankings):
        rank_df = pd.melt(scaled_rankings,value_vars=scaled_rankings.columns,ignore_index=False)
        rank_df['perturbed tf'] = rank_df.index
        rank_df = rank_df.reset_index(drop=True)
        rank_df.rename({'value':'scaled ranking'},axis=1,inplace=True)
        rank_df.rename({'variable':'regulon'},axis=1,inplace=True)

        return rank_df#, unscaled_rank_df

    def get_tfs_of_interest(self, perturbation_df):
        df_tf_of_interest = perturbation_df.copy()
        df_tf_of_interest.reset_index(inplace=True)
        pert_tfs = set(df_tf_of_interest['perturbed tf'].tolist())
        pred_tfs = set(df_tf_of_interest['regulon'].tolist())
        #df_tf_of_interest['tf'] = df_tf_of_interest.index
        tfs_of_interest = list(pert_tfs.intersection(pred_tfs))
        df_tf_of_interest = df_tf_of_interest[df_tf_of_interest['regulon'].isin(tfs_of_interest)]
        df_tf_of_interest['is tf perturbed'] = (df_tf_of_interest['regulon'] == df_tf_of_interest['perturbed tf'])
        print(df_tf_of_interest['is tf perturbed'])
        koed_tfs_df = df_tf_of_interest.loc[df_tf_of_interest['is tf perturbed'] == True]
        koed_tfs_df = koed_tfs_df.drop(['index', 'perturbed tf', 'is tf perturbed'],axis=1).T
        print(koed_tfs_df.head(10))

        return koed_tfs_df

    #def combine_samples(self):
    #    for df in self.tfs_of_interest:

    def encoder_dist(self):
        ranked = self.results.argsort()
        zscored = stats.zscore(ranked,axis=1)
        mean = np.mean(zscored,axis=0)
        std = np.std(zscored,axis=0)
        #coeffcient of variation
        CV = std/mean
        print(CV)
        with open(self.savedir+'/CV_consistency.pkl','wb+') as f:
            pkl.dump(CV, f)
        self.CV = CV
        return CV

    def plot_cv(self):
        cv_mean = np.mean(self.CV, axis=0)
        cv_std = np.std(self.CV, axis=0)
        x = np.arange(1,self.CV.shape[1]+1)
        order = cv_mean.argsort()
        cv_mean = cv_mean[order]
        cv_std = cv_std[order]
        plt.errorbar(x,cv_mean,yerr=cv_std,fmt='o')
        plt.savefig(self.savedir+'/consistency_plot.png')

    def plot_dist(self):
        plt.clf()
        plt.figure()
        for i in self.CV:
            #sns.distplot(i,hist=False,kde=True,kde_kws={'shade':True,'linewidth':3})
            plt.hist(i,bins=20,alpha=0.3,histtype='stepfilled')
        plt.savefig(self.savedir+'/consistency_cv_hist.png')

        plt.clf()
        plt.figure()
        for i in self.results:
            #sns.distplot(i,hist=False,kde=True,kde_kws={'shade':True,'linewidth':3})
            plt.hist(i,bins=20,alpha=0.3,histtype='stepfilled')
        plt.savefig(self.savedir+'/consistency_output_hist.png')

            
    #def CV_boxplot(self,CV, null_CV):
    #    plt.clf()
    #    data = np.array([CV, null_CV])
    #    labels = ['CV','null_CV']
    #    fig,ax = plt.subplots()
    #    ax.violinplot(data)
    #    ax.set_xticks(np.arange(1,len(labels)+1))
    #    ax.set_xticklabels(labels)
    #    ax.set_title('Consistency CV:'+str(np.mean(CV))+' null CV:'+str(np.mean(null_CV)))
    #    plt.savefig(self.savedir+'/CV_boxplot.png')

    #def is_significant(self):
    #    ae_mean, ae_std = self.encoder_dist()
    #    null_mean, null_std = self.null_dist()
    #    #coeffcient of variation
    #    CV = ae_std/ae_mean
    #    null_CV = null_std/null_mean
    #    standard_error = null_std/np.sqrt(self.trials)
    #    print('standard error',standard_error)
    #    z_score = (ae_mean-null_mean)/standard_error
    #    print('z score',z_score)
    #    two_sided_p_value = stats.norm(0,1).cdf(z_score)*2
    #    print('two sided p value',two_sided_p_value)
    #    mean_pval = np.mean(two_sided_p_value.flatten())
    #    self.boxplot(ae_mean,ae_std,null_mean,null_std,mean_pval)
    #    self.CV_boxplot(CV,null_CV)
    #    return two_sided_p_value
    
    #def null_dist(self):
    #    nTF = len(self.data_obj.tfs)
    #    N = self.trials
    #    tmp = np.zeros((nTF,N))
    #    for i in range(N):
    #        tmp[:, i] = np.random.permutation(nTF)

    #    rank = np.arange(nTF)
    #    print('rank',rank)
    #    mean = np.mean(tmp, axis=1)
    #    print('mean',mean)
        #order = numpy.argsort(mean)
        #spread = numpy.std(tmp, axis=1)[order]
    #    spread = np.std(tmp, axis=1)
    #    print('spread',spread)
        #mean = mean[order]
        #print(numpy.mean(spread), numpy.std(spread))
    #    return mean,spread

    #def boxplot(self,ae_mean,ae_std,null_mean,null_std,pval):
    #    plt.clf()
    #    data = np.array([ae_mean,ae_std,null_mean,null_std])
    #    labels = ['ae mean','ae_std','null mean','null std']
    #    fig,ax = plt.subplots()
    #    ax.violinplot(data)
    #    ax.set_xticks(np.arange(1,len(labels)+1))
    #    ax.set_xticklabels(labels)
    #    ax.set_title('Consistency pval:'+str(pval))
    #    plt.savefig(self.savedir+'/consistency_boxplot.png')




                     
        




