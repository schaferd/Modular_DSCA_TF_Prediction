from sklearn import metrics
import statsmodels.stats.multitest as sm
import scipy.stats
import numpy as np
import pandas as pd
import os
import sys
from pyensembl import EnsemblRelease
import seaborn as sns
ensembl_data = EnsemblRelease(78)
import matplotlib.pyplot as plt



class RNASeqTFEval():
    def __init__(self, TF_rnaseq_path, activity_path, outdir):
        self.TF_rnaseq = pd.read_pickle(TF_rnaseq_path)
        self.outdir = outdir
        self.activities = self.import_activities(activity_path)
        self.tfs = self.activities.columns
        self.tfs_ensembl = [self.convert_to_ensembl(tf) for tf in self.tfs]
        self.tfs_dict = {tf:self.convert_to_ensembl(tf) for tf in self.tfs}

        self.activities.rename(columns=self.tfs_dict,inplace=True)

        self.exp_vs_act_plot(self.activities)

    def import_activities(self,activity_path):
        df = pd.read_csv(activity_path).set_index("Unnamed: 0").T
        return df


    def filter_TFRNAseq(self):
        #keep only TFs that are in embedding
        intersection = list(set(self.tfs_ensembl).intersection(set(self.TF_rnaseq.columns)))
        return self.TF_rnaseq.loc[:, intersection]

    def convert_to_ensembl(self,g):
            ensembl_id = None
            try:
                    ensembl_id = ensembl_data.gene_ids_of_gene_name(g)[0]
            except:
                    ensembl_id = None
            return ensembl_id

    def exp_vs_act_plot(self, activities):
        ranked_activities = activities.rank(axis=1,method='min',na_option='keep',ascending=True)
        scaled_ranked_activities = (ranked_activities.T/ranked_activities.max(axis=1)).T
        scaled_ranked_activities.rename(columns=self.tfs_dict,inplace=True)
        scaled_ranked_activities = scaled_ranked_activities[self.tfs_ensembl]

        activities.rename(columns=self.tfs_dict,inplace=True)
        activities = activities[self.tfs_ensembl]

        exp = self.filter_TFRNAseq()
        exp= exp.loc[activities.index]

        print("activities",activities)
        print("exp",exp)

        save_path = self.outdir+"/rnaseqtf_figs/"
        corr_list = []
        p_vals = []

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for tf in exp.columns:
            #corr = exp[tf].corr(activities[tf]) 
            corr, p_val = scipy.stats.pearsonr(exp[tf],activities[tf])

            corr_list.append(corr)
            p_vals.append(p_val)
            plt.clf()
            plt.scatter(exp[tf],activities[tf])
            plt.xlabel("gene expression")
            plt.ylabel("pred tf activity")
            plt.title("gene exp vs. tf activity, corr: "+str(corr))
            plt.tight_layout()
            plt.savefig(save_path+tf+"_scatter.png")

        adj_pvals = sm.multipletests(p_vals,method='fdr_bh')[1]
        print("corr list",corr_list)
        print("pvals",p_vals)
        print('adj pvals',adj_pvals)
        cut_off = 0.0001
        total = 0
        length = 0

        for i in adj_pvals:
            length += 1
            if i < cut_off:
                total += 1
        percent_sig = total/length
        print(percent_sig)

        plt.clf()
        plt.hist(corr_list,bins=5)
        plt.ylabel("number of tfs")
        plt.xlabel("correlation")
        plt.title("dorothea gene exp vs. tf activity corr histogram, percent sig: "+str(round(percent_sig,2)))
        plt.tight_layout()
        plt.savefig(save_path+"histogram.png")



if __name__ == "__main__":
    TF_rnaseq_path = "/home/schaferd/ae_project/input_data_processing/tf_agg_data.pkl"
    activity_path = "/nobackup/users/schaferd/ae_project_data/hdf_gene_expression_data/train_tf_agg_data_gene_id.viper_pred.csv"
    outdir = "/nobackup/users/schaferd/ae_project_data/hdf_gene_expression_data/"
    obj = RNASeqTFEval(TF_rnaseq_path,activity_path,outdir)



