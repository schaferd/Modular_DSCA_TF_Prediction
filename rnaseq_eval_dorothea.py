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
    def __init__(self, TF_rnaseq_path, activity_path,relationships, outdir):
        self.TF_rnaseq = pd.read_pickle(TF_rnaseq_path)
        self.activities = self.import_activities(activity_path)
        self.relationships = pd.read_csv(relationships, sep='\t', low_memory=False)
        self.outdir = outdir
        self.tfs = self.activities.columns
        self.tfs_ensembl = [self.convert_to_ensembl(tf) for tf in self.tfs]
        self.tfs_dict = {tf:self.convert_to_ensembl(tf) for tf in self.tfs}
        self.self_reg_tf,self.not_self_reg_tf = self.sep_self_reulating_tfs()


        self.activities.rename(columns=self.tfs_dict,inplace=True)

        print("normal")
        self.exp_vs_act_plot(self.activities)
        
        print("self regulating tfs")
        self.exp_vs_act_plot(self.activities,tf_filter=self.self_reg_tf)

        print("not self regulating tfs")
        self.exp_vs_act_plot(self.activities,tf_filter=self.not_self_reg_tf)

        #random tests
        """
        print("normal")
        self.exp_vs_act_plot(self.activities)

        print("row random")
        input_data = self.shuffle_activities(self.activities,row_random=True)
        self.exp_vs_act_plot(input_data)

        print("col random")
        input_data = self.shuffle_activities(self.activities,col_random=True)
        self.exp_vs_act_plot(input_data)
        
        print("col and row random")
        input_data = self.shuffle_activities(self.activities,col_random=True,row_random=True)
        self.exp_vs_act_plot(input_data)
        """


    def import_activities(self,activity_path):
        input_data = pd.read_csv(activity_path).set_index("Unnamed: 0").T
        return input_data

    def shuffle_activities(self,input_data,row_random=False,col_random=False):
        if row_random:
            values = input_data.values
            values = np.apply_along_axis(np.random.permutation, 0, values)
            input_data = pd.DataFrame(data=values,index=input_data.index,columns=input_data.columns)
        if col_random:
            values = input_data.values
            values = np.apply_along_axis(np.random.permutation, 1, values)
            input_data = pd.DataFrame(data=values,index=input_data.index,columns=input_data.columns)
        return input_data



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


    def sep_self_reulating_tfs(self):
        self_regulating = set()
        all_tfs = set()
        for index, row in self.relationships.iterrows():
            all_tfs.add(row['tf'])
            if row['tf'] == row['target']:
                self_regulating.add(row['tf'])
        not_self_regulating = all_tfs-self_regulating
        self_regulating = [self.convert_to_ensembl(i) for i in self_regulating if self.convert_to_ensembl(i) is not None]
        not_self_regulating = [self.convert_to_ensembl(i) for i in not_self_regulating if self.convert_to_ensembl(i) is not None]

        return self_regulating, not_self_regulating

    def exp_vs_act_plot(self, activities,tf_filter=None):
        ranked_activities = activities.rank(axis=1,method='min',na_option='keep',ascending=True)
        scaled_ranked_activities = (ranked_activities.T/ranked_activities.max(axis=1)).T
        scaled_ranked_activities.rename(columns=self.tfs_dict,inplace=True)
        scaled_ranked_activities = scaled_ranked_activities[self.tfs_ensembl]

        activities.rename(columns=self.tfs_dict,inplace=True)
        activities = activities[self.tfs_ensembl]

        exp = self.filter_TFRNAseq()
        exp= exp.loc[activities.index]

        if tf_filter is not None:
            exp = exp.filter(tf_filter,axis=1)
            activities = activities.filter(tf_filter,axis=1)

        corr_list = []
        p_vals = []

        for tf in exp.columns:
            #corr = exp[tf].corr(activities[tf]) 
            corr, p_val = scipy.stats.pearsonr(exp[tf],activities[tf])

            corr_list.append(corr)
            p_vals.append(p_val)

        adj_pvals = sm.multipletests(p_vals,method='fdr_bh')[1]
        print("corr list",corr_list)
        print("pvals",p_vals)
        print('adj pvals',adj_pvals)




if __name__ == "__main__":
    TF_rnaseq_path = "/home/schaferd/ae_project/input_data_processing/tf_agg_data.pkl"
    activity_path = "/nobackup/users/schaferd/ae_project_data/hdf_gene_expression_data/train_tf_agg_data_gene_id.viper_pred.csv"
    outdir = "/nobackup/users/schaferd/ae_project_data/hdf_gene_expression_data/"
    relationships = "/nobackup/users/schaferd/ae_project_data/dorothea_tf_gene_relationship_knowledge/dorotheaSelectionAB_1.tsv"
    obj = RNASeqTFEval(TF_rnaseq_path,activity_path,relationships,outdir)



