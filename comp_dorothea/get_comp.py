import numpy as np
import sys
import os
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import pickle as pkl
import time
from scipy.stats.stats import pearsonr

from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)

#ae_roc_path = os.path.join(os.path.dirname(__file__),'ae_roc/')
#sys.path.append(ae_roc_path)

#diff_roc_path = os.path.join(os.path.dirname(__file__),'diff_roc/')
#sys.path.append(diff_roc_path)
import get_activity_input as gai

class getComp():
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
            self.outdir = ae_args['out_dir']

            viper_activity_dir = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/contrasts/'
            self.viper_activity_files = {'.'.join(f.split('.')[:2]):f for f in os.listdir(viper_activity_dir) if os.path.isfile('/'.join([viper_activity_dir,f])) and 'viper_pred.csv' in f}
            print(self.viper_activity_files)
            viper_activity_dir = viper_activity_dir
        #self.activities = {f:self.load_activity_file(''.join([activity_dir,self.activity_files[f]]),f)  for f in self.activity_files}
        self.viper_activities = {f:self.load_activity_file(''.join([viper_activity_dir,self.viper_activity_files[f]]),f) for f in self.viper_activity_files}
        self.aggregate_viper = self.aggregate_matrix(self.viper_activities)
        self.diff_activities = self.load_diff_activities(self.activity_file)
        self.ae_df, self.viper_df = self.filter_dfs(self.diff_activities,self.aggregate_viper)
        print(len(self.ae_df.columns), len(self.viper_df.columns))
        print(len(self.ae_df.index), len(self.viper_df.index))
        self.plot_corr(self.ae_df,self.viper_df)
        #self.plot_scatter(self.aggregate_viper,self.diff_activities)
        #self.scaled_rankings = self.rank_matrix()
        #self.perturbation_df = self.get_perturbation_info()
        #self.tfs_of_interest = self.get_tfs_of_interest()
        #self.auc = self.get_roc()

    def plot_corr(self,ae_df,viper_df):
        ae_f = ae_df.to_numpy().flatten()
        viper_f = viper_df.to_numpy().flatten()
        corr = round(pearsonr(ae_f,viper_f)[0], 3)
        plt.hist2d(x=ae_f,y=viper_f,bins=200,norm=mpl.colors.LogNorm())
        plt.xlabel('AE Embedding',size=15)
        plt.ylabel('Dorothea Embedding',size=15)
        plt.title('AE vs. Dorothea Embedding, corr: '+str(corr))
        plt.savefig(self.outdir+'/comp_corr_cycle'+str(self.cycle)+'_fold'+str(self.fold)+'.png')

    def filter_dfs(self,ae_df,viper_df):
        new_viper_cols = {col:col.split('.')[0] for col in viper_df.columns}
        viper_df.rename(columns=new_viper_cols,inplace=True)
        print(viper_df)
        v_cols = set(viper_df.columns)
        ae_cols = set(ae_df.columns)
        sim_cols = v_cols.intersection(ae_cols)

        v_rows = set(viper_df.index)
        ae_rows = set(ae_df.index)
        sim_rows = v_rows.intersection(ae_rows)

        sim_cols = list(sim_cols)
        sim_cols.sort()

        sim_rows = list(sim_rows)
        sim_rows.sort()

        ae_df = ae_df.loc[sim_rows,sim_cols]
        viper_df = viper_df.loc[sim_rows,sim_cols]

        ae_df = ae_df[~ae_df.index.duplicated(keep='first')]
        viper_df = viper_df[~viper_df.index.duplicated(keep='first')]

        ae_df = ae_df.loc[:,~ae_df.columns.duplicated(keep='first')].copy()
        viper_df = viper_df.loc[:,~viper_df.columns.duplicated(keep='first')].copy()

        return ae_df, viper_df



    def load_activity_file(self,activity_file,exp_id):
        #returns pandas df
        df = pd.read_csv(activity_file)
        df['Sample'] = exp_id
        return df

    def load_diff_activities(self,activity_file):
        #returns pandas df
        df = pd.read_csv(activity_file,index_col=0)
        #pert_tfs = list(df.index)
        return df

    def aggregate_matrix(self,activities):
        activities_list = list(activities.values())
        df = pd.concat(activities_list,ignore_index=True)
        samples = np.unique(df['Sample'].to_numpy()).tolist()
        df = pd.pivot_table(df,index=['regulon'],columns =['Sample'],values='activities') 
        return df

    def rank_matrix(self):
        pert_tfs = list(self.diff_activities.index)
        ranked_matrix = self.diff_activities.rank(axis = 1,method='min',na_option='keep',ascending=True)
        ranked_matrix = ranked_matrix.reset_index(drop=True)
        scaled_rank_matrix = (ranked_matrix.T/ranked_matrix.max(axis=1)).T
        scaled_rank_matrix.index = pert_tfs
        return scaled_rank_matrix

    def get_perturbation_info(self):
        rank_df = pd.melt(self.scaled_rankings,value_vars=self.scaled_rankings.columns,ignore_index=False)
        rank_df['perturbed tf'] = rank_df.index
        rank_df = rank_df.reset_index(drop=True)
        rank_df.rename({'value':'scaled ranking'},axis=1,inplace=True)
        rank_df.rename({'variable':'regulon'},axis=1,inplace=True)
        #activity_df = pd.melt(self.diff_activities,value_vars=self.scaled_rankings.columns,ignore_index=False)
        #activity_df.rename({'value':'pred activity'},axis=1,inplace=True)
        #rank_df['pred activity'] = activity_df['pred activity']
        #per_list = [name.split('.')[0] for name in rank_df['Sample'].tolist()]
        #rank_df['perturbed tf'] = per_list
        return rank_df

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
