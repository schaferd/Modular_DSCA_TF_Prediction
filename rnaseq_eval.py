import torch 
from sklearn import metrics
import numpy as np
import pandas as pd
import os
import sys
from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)

encoder_path = os.environ["encoder_path"]
sys.path.insert(1,encoder_path)
from encoder import AEEncoder

is_gpu = False
if torch.cuda.is_available():
    device = torch.device('cuda')
    is_gpu = True


class RNASeqTFEval():
    def __init__(self, TF_rnaseq_path, train_data, test_data, encoder, tfs):
        self.TF_rnaseq = pd.read_pickle(TF_rnaseq_path)
        self.train_data = train_data
        self.test_data = test_data 
        self.encoder = encoder.to(device)
        self.tfs = tfs
        self.tfs_ensembl = [self.convert_to_ensembl(tf) for tf in tfs]
        self.tfs_dict = {tf:self.convert_to_ensembl(tf) for tf in tfs}

        embedding = self.get_embedding(self.test_data)
        self.auc = self.ROC_Eval(embedding)
        self.corr = self.expr_activ_corr(embedding)

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

    def get_embedding(self,input_data):
        self.encoder.eval()
        samples = torch.from_numpy(np.array(input_data).astype(np.float)).to(device)
        embedding = self.encoder(samples).cpu().detach()
        embedding = pd.DataFrame(embedding,columns=self.tfs,index=input_data.index)
        return embedding


    def ROC_Eval(self,activities):
        ranked_activities = activities.rank(axis=1,method='min',na_option='keep',ascending=True)
        scaled_ranked_activities = (ranked_activities.T/ranked_activities.max(axis=1)).T
        scaled_ranked_activities.rename(columns=self.tfs_dict,inplace=True)
        scaled_ranked_activities = scaled_ranked_activities[self.tfs_ensembl]

        TF_rnaseq_filtered = self.filter_TFRNAseq()
        TF_rnaseq_filtered = TF_rnaseq_filtered.loc[scaled_ranked_activities.index]
        expected_rankings = TF_rnaseq_filtered.copy()
        expected_rankings[expected_rankings > 0] = 1
        expected_rankings[expected_rankings <= 0] = 0

        rank_df = pd.melt(scaled_ranked_activities,value_vars=scaled_ranked_activities.columns,ignore_index=False)
        rank_df.reset_index(inplace=True)
        rank_df.rename(columns={'index':'sample_id','value':'rank'},inplace=True)
        exp_df = pd.melt(expected_rankings,value_vars=expected_rankings.columns,ignore_index=False)
        exp_df.reset_index(inplace=True)
        exp_df.rename(columns={'index':'sample_id','value':'exp_rank'},inplace=True)


        combined = pd.merge(exp_df,rank_df,how='inner',on=['sample_id','variable'])

        observed = combined['rank']
        expected = combined['exp_rank']

        n_pos = sum(expected == 1)
        n_neg = sum(expected == 0)
        pos = observed[expected == 1]
        neg = observed[expected == 0]

        auc, fpr, tpr = self.get_aucROC(neg.tolist(),pos.tolist())
        print('auc', auc)

        return auc

    def expr_activ_corr(self,activities):
        activities.rename(columns=self.tfs_dict,inplace=True)
        activities = activities[self.tfs_ensembl]

        TF_rnaseq_filtered = self.filter_TFRNAseq()
        TF_rnaseq_filtered = TF_rnaseq_filtered.loc[activities.index]

        act_df = pd.melt(activities,value_vars=activities.columns,ignore_index=False)
        act_df.reset_index(inplace=True)
        act_df.rename(columns={'index':'sample_id','value':'activity'},inplace=True)
        exp_df = pd.melt(TF_rnaseq_filtered,value_vars=TF_rnaseq_filtered.columns,ignore_index=False)
        exp_df.reset_index(inplace=True)
        exp_df.rename(columns={'index':'sample_id','value':'exp'},inplace=True)

        combined = pd.merge(exp_df,act_df,how='inner',on=['sample_id','variable'])
        pd.set_option('display.max_columns', None)
        print('combined',combined)

        corr = combined['exp'].corr(combined['activity'])
        print('corr',str(corr))
        return corr


    def get_aucROC(self,ne, po):
        target = [0 for i in range(len(ne))]
        for i in range(len(po)):
            target.append(1)
        obs = ne+po
        fpr,tpr,thresholds = metrics.roc_curve(target,obs)
        auc = metrics.roc_auc_score(target,obs)
        return auc,fpr,tpr



        



