import numpy as np
import sys
import os
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import torch
import pickle as pkl
import time

from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)

#ae_roc_path = os.path.join(os.path.dirname(__file__),'ae_roc/')
#sys.path.append(ae_roc_path)

#diff_roc_path = os.path.join(os.path.dirname(__file__),'diff_roc/')
#sys.path.append(diff_roc_path)
import get_activity_input as gai

class getROCCurve():
    def __init__(self,ae_args={}):
        self.ae_args = ae_args 
        self.activity_files = None
        activity_dir = None
        self.fold = None
        self.cycle = None
        if len(self.ae_args.keys()) > 0:
            obj = gai.ActivityInput(ae_args['embedding'],ae_args['data_dir'],ae_args['knowledge'],ae_args['overlap_genes'],ae_args['ae_input_genes'],ae_args['tf_list'],ae_args['out_dir'])
            obj.get_activities()
            #self.activity_files = {'.'.join(f.split('.')[:2]):f for f in os.listdir(obj.save_path) if os.path.isfile('/'.join([obj.save_path,f])) and 'diff_activities' in f}
            self.activity_file = obj.out_dir+'/diff_activities.csv'
            activity_dir = obj.out_dir
            self.fold = ae_args['fold']
            self.cycle = ae_args['cycle']
        else:
            viper_activity_dir = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/contrasts/'
            self.activity_files = {'.'.join(f.split('.')[:2]):f for f in os.listdir(viper_activity_dir) if os.path.isfile('/'.join([viper_activity_dir,f])) and 'viper_pred.csv' in f}
            activity_dir = viper_activity_dir
        #self.activities = {f:self.load_activity_file(''.join([activity_dir,self.activity_files[f]]),f)  for f in self.activity_files}
        self.diff_activities = self.load_diff_activities(self.activity_file)
        with open(obj.out_dir+'/ko_tf_index.pkl','rb') as f:
            self.index_to_ko_tfs = pkl.load(f)
        self.scaled_rankings,self.rankings = self.rank_matrix()
        #self.perturbation_df,self.unscaled_rank_df = self.get_perturbation_info()
        self.perturbation_df = self.get_perturbation_info()
        self.tfs_of_interest = self.get_tfs_of_interest()
        self.auc = self.get_roc()


    def load_diff_activities(self,activity_file):
        #returns pandas df
        df = pd.read_csv(activity_file,index_col=0)
        #pert_tfs = list(df.index)
        return df

    """
    def aggregate_matrix(self):
        activities_list = list(self.activities.values())
        df = pd.concat(activities_list,ignore_index=True)
        samples = np.unique(df['Sample'].to_numpy()).tolist()
        df = pd.pivot_table(df,index=['regulon'],columns =['Sample'],values='activities') 
        return df
    """

    def rank_matrix(self):
        pert_tfs = list(self.diff_activities.index)
        ranked_matrix = self.diff_activities.rank(axis = 1,method='min',na_option='keep',ascending=True)
        print('tfs', len(self.diff_activities.columns))
        ranked_matrix = ranked_matrix.reset_index(drop=True)
        scaled_rank_matrix = (ranked_matrix.T/ranked_matrix.max(axis=1)).T
        scaled_rank_matrix.index = pert_tfs
        print('scaled rank matrix')
        print(scaled_rank_matrix)
        return scaled_rank_matrix, ranked_matrix

    def get_perturbation_info(self):
        rank_df = pd.melt(self.scaled_rankings,value_vars=self.scaled_rankings.columns,ignore_index=False)
        rank_df['perturbed tf'] = [self.index_to_ko_tfs[i] for i in rank_df.index]
        rank_df = rank_df.reset_index(drop=True)
        rank_df.rename({'value':'scaled ranking'},axis=1,inplace=True)
        rank_df.rename({'variable':'regulon'},axis=1,inplace=True)

        #unscaled_rank_df = pd.melt(self.rankings,value_vars=self.scaled_rankings.columns,ignore_index=False)
        #unscaled_rank_df['perturbed tf'] = unscaled_rank_df.index
        #unscaled_rank_df = unscaled_rank_df.reset_index(drop=True)
        #unscaled_rank_df.rename({'value':'scaled ranking'},axis=1,inplace=True)
        #unscaled_rank_df.rename({'variable':'regulon'},axis=1,inplace=True)
        #activity_df = pd.melt(self.diff_activities,value_vars=self.scaled_rankings.columns,ignore_index=False)
        #activity_df.rename({'value':'pred activity'},axis=1,inplace=True)
        #rank_df['pred activity'] = activity_df['pred activity']
        #per_list = [name.split('.')[0] for name in rank_df['Sample'].tolist()]
        #rank_df['perturbed tf'] = per_list
        return rank_df#, unscaled_rank_df

    def get_tfs_of_interest(self):
        df_tf_of_interest = self.perturbation_df.copy()
        df_tf_of_interest.reset_index(inplace=True)
        pert_tfs = set(df_tf_of_interest['perturbed tf'].tolist())
        pred_tfs = set(df_tf_of_interest['regulon'].tolist())
        #df_tf_of_interest['tf'] = df_tf_of_interest.index
        tfs_of_interest = list(pert_tfs.intersection(pred_tfs))
        df_tf_of_interest = df_tf_of_interest[df_tf_of_interest['regulon'].isin(tfs_of_interest)]
        df_tf_of_interest['is tf perturbed'] = (df_tf_of_interest['regulon'] == df_tf_of_interest['perturbed tf'])
        print(df_tf_of_interest['is tf perturbed'])
        print('koed tfs df')
        koed_tfs_df = df_tf_of_interest.loc[df_tf_of_interest['is tf perturbed'] == True]
        print(koed_tfs_df)
        return df_tf_of_interest

    def get_roc(self):
        observed = self.tfs_of_interest['scaled ranking']
        expected = self.tfs_of_interest['is tf perturbed']+0

        n_positives = sum(expected == 1)
        n_negatives = sum(expected == 0)
        positives = observed[expected == 1]
        negatives = observed[expected == 0]
        #n = min(n_positives,n_negatives)
        #r_positives = [positives.sample(n,replace=False).tolist() for i in range(100)]
        #r_negatives = [negatives.sample(n,replace=False).tolist() for i in range(100)]
        #print('positives')
        #print(r_positives)
        #for i in range(len(r_positives)):
        #    print(r_positives[i])

        auc,fpr,tpr = self.get_aucROC(negatives.tolist(),positives.tolist())

        print("auc")
        print(auc)
        print("fpr")
        print(fpr)
        print("tpr")
        print(tpr)

        self.plot_ROC(tpr,fpr,auc)
        return auc


    def get_aucROC(self,ne, po):
        target = [0 for i in range(len(ne))]
        for i in range(len(po)):
            target.append(1)
        obs = ne+po
        fpr,tpr,thresholds = metrics.roc_curve(target,obs)
        auc = metrics.roc_auc_score(target,obs)
        return auc,fpr,tpr

    def plot_ROC(self,tpr,fpr,auc):
        plt.plot(fpr,tpr,color="darkorange",label="ROC Curve (area = %0.2f)"%auc)
        plt.plot([0,1],[0,1],color="navy",linestyle="--")
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (area = %0.2f)"%auc)
        if len(self.ae_args.keys()) > 0:
            plt.savefig(self.ae_args['out_dir']+"/roc_ae_cycle_"+str(self.cycle)+"_fold"+str(self.fold)+".png")
        else:
            plt.savefig('roc_viper.png')


def open_pkl(pkl_file):
        with open(pkl_file,'rb') as f:
                return pkl.load(f)

def convert_gene_name_to_ensembl(gene_name):
    try:
        ensembl_id = ensembl_data.gene_ids_of_gene_name(gene_name)
        return ensembl_id
    except:
        return [None]

def get_knowledge(knowledge_path,overlap_set):
    df = pd.read_csv(knowledge_path,sep='\t',low_memory=False)
    tf_gene_dict = {}
    for i,row in df.iterrows():
        translated_target = convert_gene_name_to_ensembl(row['target'])[0]
        if translated_target is not None:
            if row['tf'] in tf_gene_dict and translated_target in overlap_set:
                tf_gene_dict[row['tf']].append(row['target'])
            elif translated_target in overlap_set:
                tf_gene_dict[row['tf']] = [row['target']]
    return tf_gene_dict
        

if __name__ == '__main__':
    embedding_path = '/nobackup/users/schaferd/ae_project_outputs/vanilla/moa_tests_epochs100_batchsize256_edepth2_ddepth2_lr1e-05_lrsched_moa0.1_7-19_18.36.45/model_encoder_fold0.pth'
    data_dir = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/contrasts/'
    knowledge_path = '/nobackup/users/schaferd/ae_project_data/dorothea_tf_gene_relationship_knowledge/dorotheaSelectionA.tsv'
    overlap_genes_path = '/nobackup/users/schaferd/ko_eval_data/ae_data/overlap_list.pkl'
    ae_input_genes_path = '/nobackup/users/schaferd/ko_eval_data/ae_data/input_genes.pkl'
    out_dir = '/'.join(embedding_path.split('/')[:-1]) 
    tf_list_path = out_dir+'/tfs_in_embedding.pkl'#'/nobackup/users/schaferd/ko_eval_data/ae_data/embedding_tf_names.pkl'

    embedding = torch.load(embedding_path)
    overlap_genes = open_pkl(overlap_genes_path)
    knowledge = get_knowledge(knowledge_path,set(overlap_genes))
    ae_input_genes = open_pkl(ae_input_genes_path)
    tf_list = open_pkl(tf_list_path)

    ae_args = {
        'embedding' :embedding,
        'overlap_genes': overlap_genes,
        'knowledge':knowledge,
        'data_dir':data_dir,
        'ae_input_genes':ae_input_genes,
        'tf_list':tf_list,
        'out_dir':out_dir,
        'fold':0
        }
    obj = getROCCurve(ae_args=ae_args)
