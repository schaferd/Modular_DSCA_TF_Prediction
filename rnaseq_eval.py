import torch 
import statsmodels.stats.multitest as sm
import scipy.stats
from sklearn import metrics
import numpy as np
import pandas as pd
import os
import sys
from pyensembl import EnsemblRelease
import seaborn as sns
ensembl_data = EnsemblRelease(78)
import matplotlib.pyplot as plt

encoder_path = os.environ["encoder_path"]
sys.path.insert(1,encoder_path)
from encoder import AEEncoder

is_gpu = False
if torch.cuda.is_available():
    device = torch.device('cuda')
    is_gpu = True


class RNASeqTFEval():
    def __init__(self, TF_rnaseq_path, train_data, test_data, encoder, tfs, relationships, outdir):
        self.TF_rnaseq = pd.read_pickle(TF_rnaseq_path)
        self.train_data = train_data
        self.relationships = pd.read_csv(relationships, sep='\t', low_memory=False)
        self.test_data = test_data 
        self.encoder = encoder.to(device)
        self.tfs = tfs
        self.tfs_ensembl = [self.convert_to_ensembl(tf) for tf in tfs]
        self.tfs_dict = {tf:self.convert_to_ensembl(tf) for tf in tfs}
        self.outdir = outdir
        self.self_reg_tf,self.not_self_reg_tf = self.sep_self_reulating_tfs()

        self.run_self_reg_tests()
        print("---------------------------zscored tests----------------------------")
        self.run_random_tests(is_zscored=True)

    def run_self_reg_tests(self):
        print("normal")
        embedding = self.get_embedding(self.test_data)
        self.exp_vs_act_plot(embedding)
        
        print("self regulating tfs")
        embedding = self.get_embedding(self.test_data)
        self.exp_vs_act_plot(embedding,tf_filter=self.self_reg_tf)

        print("not self regulating tfs")
        embedding = self.get_embedding(self.test_data)
        self.exp_vs_act_plot(embedding,tf_filter=self.not_self_reg_tf)


    def run_random_tests(self,is_zscored=False):
        print("normal")
        embedding = self.get_embedding(self.test_data,zscore=is_zscored)
        self.exp_vs_act_plot(embedding)

        print("row random")
        embedding = self.get_embedding(self.test_data,row_random=True,zscore=is_zscored)
        self.exp_vs_act_plot(embedding)

        print("col random")
        embedding = self.get_embedding(self.test_data,col_random=True,zscore=is_zscored)
        self.exp_vs_act_plot(embedding)
        
        print("col and row random")
        embedding = self.get_embedding(self.test_data,col_random=True, row_random=True,zscore=is_zscored)
        self.exp_vs_act_plot(embedding)

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

    def get_embedding(self,input_data,row_random=False,col_random=False,zscore=False):
        if row_random:
            values = input_data.values
            values = np.apply_along_axis(np.random.permutation, 0, values)
            input_data = pd.DataFrame(data=values,index=input_data.index,columns=input_data.columns)
        if col_random:
            values = input_data.values
            values = np.apply_along_axis(np.random.permutation, 1, values)
            input_data = pd.DataFrame(data=values,index=input_data.index,columns=input_data.columns)
        self.encoder.eval()
        samples = torch.from_numpy(np.array(input_data).astype(np.float)).to(device)
        embedding = self.encoder(samples).cpu().detach()
        embedding = pd.DataFrame(embedding,columns=self.tfs,index=input_data.index)
        if zscore:
            embedding = ((embedding.T-embedding.T.mean())/embedding.T.std()).T
        return embedding

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
        pvals = []

        for tf in exp.columns:
            corr, p_val = scipy.stats.pearsonr(exp[tf],activities[tf])
            corr_list.append(corr)
            pvals.append(p_val)

        adj_pvals = sm.multipletests(pvals,method='fdr_bh')[1]
        print("corr list",corr_list)
        print("pvals",pvals)
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

        





