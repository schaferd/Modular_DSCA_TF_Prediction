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

    def __init__(self, encoder_dir, data_obj,trials=5):
        print('encoder dir',encoder_dir)
        print(os.listdir(encoder_dir))
        self.savedir = encoder_dir
        self.data_obj = data_obj
        self.input_data = data_obj.input_data
        self.sample_rows = self.input_data.iloc[sample(range(len(self.input_data.index)),trials),:]
        self.input_tensor = torch.from_numpy(np.array(self.sample_rows).astype(np.float)).to(device).float()
        self.trials = trials
        self.results = None
        self.CV = None
        
    def get_output(self,encoder):
        encoder.eval()
        tfs = encoder(self.input_tensor).cpu().detach().numpy()
        print('tfs consistency',tfs)
        if self.results is None:
            self.results = np.array([tfs])
        else:
            self.results = np.append(self.results,[tfs],axis=0) 

    def save_output_data(self,encoder,whole_ae):
        encoder.eval()
        whole_ae.eval()
        print(self.input_data)
        filtered_input_data = self.input_data.loc[:,self.data_obj.overlapping_genes]
        tfs = encoder(self.input_tensor)
        output = whole_ae(tfs).cpu().detach().numpy()
        tfs = tfs.cpu().detach().numpy()
        with open(self.savedir+'/input_data.pkl','wb+') as f:
            pkl.dump(self.input_tensor.cpu().detach().numpy(),f)

        with open(self.savedir+'/tfs_data.pkl','wb+') as f:
            pkl.dump(tfs,f)

        with open(self.savedir+'/output_data.pkl','wb+') as f:
            pkl.dump(output,f)

    def make_random_cv(self):
        rand = np.random.rand(*self.results.shape)
        ranked = rand.argsort()
        mean = np.mean(ranked,axis=0)
        std = np.std(ranked,axis=0)
        rand_CV=std/mean
        with open(self.savedir+'/rand_CV_consistency.pkl','wb+') as f:
            pkl.dump(rand_CV, f)


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




                     
        





