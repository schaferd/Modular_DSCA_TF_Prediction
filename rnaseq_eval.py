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
    def __init__(self, TF_rnaseq_path, train_data, test_data, encoder, tfs, outdir):
        self.TF_rnaseq = pd.read_pickle(TF_rnaseq_path)
        self.train_data = train_data
        self.test_data = test_data 
        self.encoder = encoder.to(device)
        self.tfs = tfs
        self.tfs_ensembl = [self.convert_to_ensembl(tf) for tf in tfs]
        self.tfs_dict = {tf:self.convert_to_ensembl(tf) for tf in tfs}
        self.outdir = outdir

        embedding = self.get_embedding(self.test_data)
        self.auc= 0
        max_act_cutoff = 0
        max_exp_cutoff = 0
        step = 0.2
        act_range = np.arange(0,1+step,step)
        exp_range = np.arange(0,1+step,step)
        self.exp_vs_act_plot(embedding)
        embedding = self.get_embedding(self.test_data,is_random=True)
        self.exp_vs_act_plot(embedding,is_random=True)

        """
        fprs, tprs = [],[]
        for i in act_range:
            tpr,fpr = self.ROC_Eval(embedding,i,i)
            fprs.append(fpr)
            tprs.append(tpr)
        print("fpr",fprs,"tprs",tprs)
        self.auc = metrics.auc(fprs,tprs)
        print("auc",str(self.auc))

        aucs = []
        aucs2 = []
        for i in act_range:
            fprs, tprs = [],[]
            fprs2, tprs2 = [],[]
            for j in exp_range:
                tpr,fpr = self.ROC_Eval(embedding,i,j)
                tpr2,fpr2 = self.ROC_Eval_random(embedding,i,j)
                fprs.append(fpr)
                tprs.append(tpr)
                fprs2.append(fpr2)
                tprs2.append(tpr2)
            aucs.append(metrics.auc(fprs,tprs))
            aucs2.append(metrics.auc(fprs2,tprs2))
        #self.auc = metrics.auc(act_range,aucs)
        print("aucs",str(aucs))
        print("aucs2",str(aucs2))

        self.corr = self.expr_activ_corr(embedding)
        """

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

    def get_embedding(self,input_data,is_random=False):
        if is_random:
            values = input_data.values
            values = np.apply_along_axis(np.random.permutation, 0, values)
            values = np.apply_along_axis(np.random.permutation, 1, values)
            input_data = pd.DataFrame(data=values,index=input_data.index,columns=input_data.columns)
        self.encoder.eval()
        samples = torch.from_numpy(np.array(input_data).astype(np.float)).to(device)
        embedding = self.encoder(samples).cpu().detach()
        embedding = pd.DataFrame(embedding,columns=self.tfs,index=input_data.index)
        return embedding


    def exp_vs_act_plot(self, activities, is_random=False):
        ranked_activities = activities.rank(axis=1,method='min',na_option='keep',ascending=True)
        scaled_ranked_activities = (ranked_activities.T/ranked_activities.max(axis=1)).T
        scaled_ranked_activities.rename(columns=self.tfs_dict,inplace=True)
        scaled_ranked_activities = scaled_ranked_activities[self.tfs_ensembl]

        activities.rename(columns=self.tfs_dict,inplace=True)
        activities = activities[self.tfs_ensembl]

        exp = self.filter_TFRNAseq()
        if is_random:
            values = exp.values
            values = np.apply_along_axis(np.random.permutation, 0, values)
            values = np.apply_along_axis(np.random.permutation, 1, values)
            exp = pd.DataFrame(data=values,index=exp.index,columns=exp.columns)

        exp= exp.loc[activities.index]

        print("activities",activities)
        print("exp",exp)

        save_path = self.outdir+"/rnaseqtf_figs/"
        if is_random:
            save_path = self.outdir+"/rnaseqtf_random_figs/"
        corr_list = []
        pvals = []

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for tf in exp.columns:
            #corr = exp[tf].corr(activities[tf]) 
            corr, p_val = scipy.stats.pearsonr(exp[tf],activities[tf])
            corr_list.append(corr)
            pvals.append(p_val)

            """
            plt.clf()
            plt.scatter(exp[tf],activities[tf])
            plt.xlabel("gene expression")
            plt.ylabel("pred tf activity")
            plt.title("gene exp vs. tf activity, corr: "+str(corr))
            plt.tight_layout()
            plt.savefig(save_path+tf+"_scatter.png")
            """

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

        plt.clf()
        plt.hist(corr_list,bins=5)
        plt.ylabel("number of tfs")
        plt.xlabel("correlation")
        plt.title("ae gene exp vs. tf activity corr histogram, pval: "+str(round(percent_sig,2)))
        plt.tight_layout()
        plt.savefig(save_path+"histogram.png")

        


    def ROC_Eval(self,activities,act_cutoff,exp_cutoff):

        ranked_activities = activities.rank(axis=1,method='min',na_option='keep',ascending=True)
        scaled_ranked_activities = (ranked_activities.T/ranked_activities.max(axis=1)).T
        scaled_ranked_activities.rename(columns=self.tfs_dict,inplace=True)
        scaled_ranked_activities = scaled_ranked_activities[self.tfs_ensembl]

        activities.rename(columns=self.tfs_dict,inplace=True)
        activities = activities[self.tfs_ensembl]

        activities[scaled_ranked_activities <= act_cutoff] = 0
        activities[scaled_ranked_activities > act_cutoff] = 1
        
        TF_rnaseq_filtered = self.filter_TFRNAseq()
        TF_rnaseq_filtered = TF_rnaseq_filtered.loc[activities.index]
        expected_rankings = TF_rnaseq_filtered.rank(axis=1,method='min',na_option='keep',ascending=True)
        expected_rankings = (expected_rankings.T/expected_rankings.max(axis=1)).T
        expected_rankings[expected_rankings > exp_cutoff] = 1
        expected_rankings[expected_rankings <= exp_cutoff] = 0

        rank_df = pd.melt(scaled_ranked_activities,value_vars=scaled_ranked_activities.columns,ignore_index=False)
        rank_df.reset_index(inplace=True)
        rank_df.rename(columns={'index':'sample_id','value':'rank'},inplace=True)

        act_df = pd.melt(activities,value_vars=activities.columns,ignore_index=False)
        act_df.reset_index(inplace=True)
        act_df.rename(columns={'index':'sample_id','value':'is_on'},inplace=True)

        exp_df = pd.melt(expected_rankings,value_vars=expected_rankings.columns,ignore_index=False)
        exp_df.reset_index(inplace=True)
        exp_df.rename(columns={'index':'sample_id','value':'exp_on'},inplace=True)

        combined = pd.merge(exp_df,act_df,how='inner',on=['sample_id','variable'])
        print("combined",combined)
        f = combined[combined["exp_on"]==1]
        fp = f[f["is_on"]==0]
        fn = f[f["is_on"]==1]
        
        t = combined[combined["exp_on"]==0]
        tp = t[t["is_on"] == 0]
        tn = t[t["is_on"] == 1]

        tp_fn = len(tp.index)+len(fn.index)
        fp_tn = len(fp.index)+len(tn.index)

        print("tp_fn",tp_fn)
        print("fp_tn",fp_tn)

        tpr = 0
        fpr = 0

        if tp_fn > 0:
            tpr = len(tp.index)/tp_fn
        if fp_tn > 0:
            fpr = len(fp.index)/fp_tn

        print("tpr",tpr,"fpr",fpr)
        return tpr,fpr

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



    def ROC_Eval_random(self,activities,act_cutoff,exp_cutoff):

        activities = pd.DataFrame(np.random.randint(0,100,size=(len(activities.index),len(activities.columns))),index=activities.index,columns = activities.columns)

        ranked_activities = activities.rank(axis=1,method='min',na_option='keep',ascending=True)
        scaled_ranked_activities = (ranked_activities.T/ranked_activities.max(axis=1)).T
        scaled_ranked_activities.rename(columns=self.tfs_dict,inplace=True)
        scaled_ranked_activities = scaled_ranked_activities[self.tfs_ensembl]

        activities.rename(columns=self.tfs_dict,inplace=True)
        activities = activities[self.tfs_ensembl]

        activities[scaled_ranked_activities <= act_cutoff] = 0
        activities[scaled_ranked_activities > act_cutoff] = 1
        
        TF_rnaseq_filtered = self.filter_TFRNAseq()
        TF_rnaseq_filtered = TF_rnaseq_filtered.loc[activities.index]
        expected_rankings = TF_rnaseq_filtered.rank(axis=1,method='min',na_option='keep',ascending=True)
        expected_rankings = (expected_rankings.T/expected_rankings.max(axis=1)).T
        expected_rankings[expected_rankings > exp_cutoff] = 1
        expected_rankings[expected_rankings <= exp_cutoff] = 0

        rank_df = pd.melt(scaled_ranked_activities,value_vars=scaled_ranked_activities.columns,ignore_index=False)
        rank_df.reset_index(inplace=True)
        rank_df.rename(columns={'index':'sample_id','value':'rank'},inplace=True)

        act_df = pd.melt(activities,value_vars=activities.columns,ignore_index=False)
        act_df.reset_index(inplace=True)
        act_df.rename(columns={'index':'sample_id','value':'is_on'},inplace=True)

        exp_df = pd.melt(expected_rankings,value_vars=expected_rankings.columns,ignore_index=False)
        exp_df.reset_index(inplace=True)
        exp_df.rename(columns={'index':'sample_id','value':'exp_on'},inplace=True)

        combined = pd.merge(exp_df,act_df,how='inner',on=['sample_id','variable'])
        print("combined",combined)
        f = combined[combined["exp_on"]==1]
        fp = f[f["is_on"]==0]
        fn = f[f["is_on"]==1]
        
        t = combined[combined["exp_on"]==0]
        tp = t[t["is_on"] == 0]
        tn = t[t["is_on"] == 1]

        tp_fn = len(tp.index)+len(fn.index)
        fp_tn = len(fp.index)+len(tn.index)

        print("tp_fn",tp_fn)
        print("fp_tn",fp_tn)

        tpr = 0
        fpr = 0

        if tp_fn > 0:
            tpr = len(tp.index)/tp_fn
        if fp_tn > 0:
            fpr = len(fp.index)/fp_tn

        print("tpr",tpr,"fpr",fpr)
        return tpr,fpr
        



