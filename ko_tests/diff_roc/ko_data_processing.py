import pandas as pd
import numpy as np
import torch
import os
import sys
import pickle as pkl
from scipy import stats
from scipy.stats import zscore
import time
from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)

class ActivityInput():
    def __init__(self,data_dir,ae_input_genes):
        self.data_dir = data_dir
        self.negative_samples = [data_dir+f for f in os.listdir(data_dir) if os.path.isfile(data_dir+f) and 'negative' in f]
        self.positive_samples = [data_dir+f for f in os.listdir(data_dir) if os.path.isfile(data_dir+f) and 'positive' in f]

        self.negative_samples.sort()
        self.positive_samples.sort()

        pos_df,neg_df = self.get_pos_neg_matrices()
        pos_df.to_csv(self.data_dir+'/pos_df.csv')
        neg_df.to_csv(self.data_dir+'/neg_df.csv')



    def get_pos_neg_matrices(self):
        pos_df = None
        neg_df = None
        for i,pos in enumerate(self.positive_samples):
            neg = self.get_sample(self.negative_samples[i])
            pos = self.get_sample(pos)
            if pos_df is None:
                pos_df = pos
                neg_df = neg
            else:
                pos_df = pd.concat([pos_df,pos],ignore_index=False) #ignore_index=True, join='inner')
                neg_df = pd.concat([neg_df,neg],ignore_index=False) #ignore_index=True,join='inner')

        pos_df = pos_df.fillna(0)
        neg_df = neg_df.fillna(0)

        pos_neg_df = pd.concat([pos_df,neg_df])
        pos_neg_df = pos_neg_df.apply(zscore)

        pos_df = pos_neg_df.iloc[:len(pos_df.index)]
        neg_df = pos_neg_df.iloc[len(pos_df.index):]

        print(pos_df,neg_df)
        return pos_df, neg_df


    def get_sample(self,data_path):
        data = self.get_list_from_file(data_path)
        exp = data_path.split('/')[-1].split('_')[1]
        pert_tf = data_path.split('/')[-1].split('_')[0]
        gene_expression_dict = {}
        gene_list = []
        gene_set = set()
        for i,row in enumerate(data):
            temp = row.split(' ')
            if len(temp) > 2 and temp[1] != 'NA' and temp[-1].strip() != 'NA':
                for i in temp[1:-1]:
                    gene = i.replace('"','').replace('/','')
                    genes = self.convert_gene_name_to_ensembl(gene)
                    if genes[0] is not None: 
                        for g in genes:
                            if g not in gene_set:
                                gene_list.append(g)
                                gene_expression_dict[g] = float(temp[-1].strip())
                                gene_set.add(g)
        
        ordered_exp = [gene_expression_dict[g] for g in gene_list]
        df = pd.DataFrame([ordered_exp],columns=gene_list,index=[exp+'_'+pert_tf])
        #df = df.apply(stats.zscore,axis=1)
        return df

    def convert_gene_name_to_ensembl(self,gene_name):
        try:
            ensembl_id = ensembl_data.gene_ids_of_gene_name(gene_name)
            return ensembl_id
        except:
            return [None]

    def get_list_from_file(self,f):
        file_list = []
        with open(f,'r') as f_:
            lines = f_.readlines()
            for line in lines:
                file_list.append(line)
        return file_list



if __name__ == '__main__':
    #def __init__(self,embedding_path,data_dir,knowledge_path,overlap_genes_path,ae_input_genes,tf_list_path):
    #embedding_path = '/nobackup/users/schaferd/ae_project_outputs/vanilla/get_model_info_epochs3_batchsize128_edepth2_ddepth2_lr0.0001_6-23_13.31.0/model_encoder_fold0.pth'
    data_dir = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/pos_neg_samples/'
    ae_input_genes = '/nobackup/users/schaferd/ko_eval_data/ae_data/input_genes.pkl'
    tf_list_path = '/nobackup/users/schaferd/ko_eval_data/ae_data/embedding_tf_names.pkl'
    obj = ActivityInput(data_dir,ae_input_genes)

