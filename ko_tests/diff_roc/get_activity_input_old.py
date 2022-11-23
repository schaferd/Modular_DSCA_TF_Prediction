import pandas as pd
import numpy as np
import torch
import os
import sys
import pickle as pkl
from scipy import stats
import time
from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)

encoder_path = os.environ["encoder_path"]
sys.path.insert(1,encoder_path)
from encoder import AEEncoder

is_gpu = False
device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
    is_gpu = True
print("is gpu 1"+str(is_gpu))


class ActivityInput():
    def __init__(self,encoder,data_dir,knowledge,overlap_genes,ae_input_genes,tf_list,out_dir):
        self.data_dir = data_dir
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.negative_samples = [data_dir+f for f in os.listdir(data_dir) if os.path.isfile(data_dir+f) and 'negative' in f]
        self.positive_samples = [data_dir+f for f in os.listdir(data_dir) if os.path.isfile(data_dir+f) and 'positive' in f]

        self.negative_samples.sort()
        self.positive_samples.sort()

        self.overlap_genes = overlap_genes
        self.tf_list = tf_list
        self.out_dir = out_dir
        self.ae_input_genes = ae_input_genes
        self.knowledge = knowledge
        self.get_activities()
        #for i, pos_sample in enumerate(self.positive_samples):
        #    self.get_activities(pos_sample,self.negative_samples[i])

    def get_activities(self):
        pos_matrix, neg_matrix = self.get_pos_neg_matrices()
        pos_embedding = self.encoder(pos_matrix).cpu().detach().numpy()
        neg_embedding = self.encoder(neg_matrix).cpu().detach().numpy()
        diff_embedding = pos_embedding-neg_embedding
        pert_tfs = [f.split('/')[-1].split('_')[0] for f in self.positive_samples]
        print('diff embedding',diff_embedding)
        diff_df = pd.DataFrame(data=diff_embedding,index=pert_tfs,columns=self.tf_list)
        print('diff df',diff_df)
        diff_df.to_csv(self.out_dir+'/diff_activities.csv')


    """
    def get_activities(self,pos_path,neg_path):
        pos_sample = torch.from_numpy(np.array(self.get_sample(pos_path)).astype(np.float)).double().to(device)
        neg_sample = torch.from_numpy(np.array(self.get_sample(neg_path)).astype(np.float)).double().to(device)
        print('pos sample',pos_sample)
        print('neg sample',neg_sample)
        pos_embedding = self.encoder(pos_sample).cpu().detach().numpy()
        neg_embedding = self.encoder(neg_sample).cpu().detach().numpy()
        diff_embedding = pos_embedding - neg_embedding
        size = self.get_size(pos_sample)
        csv_file = '_'.join(pos_path.split('/')[-1].split('.')[0].split('_')[:-1])+'.diff_activities.csv'
        csv_path = self.save_path+'/'+csv_file
        data_struct = {'regulon':self.tf_list,'activities':embedding[0],'size':size}
        df = pd.DataFrame(data_struct)
        df.to_csv(csv_path)
    """

    def get_pos_neg_matrices(self):
        pos_df = pd.DataFrame(columns = self.ae_input_genes)
        neg_df = pd.DataFrame(columns = self.ae_input_genes)
        for i,pos in enumerate(self.positive_samples):
            neg = self.get_sample(self.negative_samples[i])
            pos = self.get_sample(pos)
            pos_df = pd.concat([pos_df,pos], ignore_index=True)
            neg_df = pd.concat([neg_df,neg], ignore_index=True)
        pos_matrix = torch.from_numpy(np.array(pos_df).astype(np.float)).to(device).float()
        neg_matrix = torch.from_numpy(np.array(neg_df).astype(np.float)).to(device).float()
        return pos_matrix, neg_matrix 


    def get_sample(self,data_path):
        data = self.get_list_from_file(data_path)
        gene_expression_dict = {}
        gene_list = []
        for i,row in enumerate(data):
            temp = row.split(' ')
            if len(temp) > 2 and temp[1] != 'NA' and temp[-1].strip() != 'NA':
                gene = temp[1].replace('"','')
                genes = self.convert_gene_name_to_ensembl(gene)
                if genes[0] is not None: 
                    gene_list.extend(genes)
                    gene_expression_dict[genes[0]] = float(temp[-1].strip())
        
        df = pd.DataFrame(gene_expression_dict,index=[0])
        overlapping_genes = self.get_overlapping_genes(df.columns)
        ordered_exp = [gene_expression_dict[gene] for gene in overlapping_genes]
        #ordered_exp = stats.zscore(ordered_exp)
        
        input_df = pd.DataFrame(columns = self.ae_input_genes)
        df0 = pd.DataFrame(0,index=[len(input_df.index)],columns=input_df.columns)
        input_df = pd.concat([input_df,df0], ignore_index=True)
        input_df.loc[0,overlapping_genes] = ordered_exp 
    
        return input_df

    def get_size(self,sample):
        size_dict = {}
        size_list = []
        sample_genes = set(sample.columns[1:])
        for tf in self.tf_list:
            tf_genes = set([self.convert_gene_name_to_ensembl(gene)[0] for gene in self.knowledge[tf]])
            genes = tf_genes.intersection(sample_genes)
            size_dict[tf] = len(genes)
            size_list.append(len(genes))
        return size_list

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

    def get_overlapping_genes(self,genes):
        overlapping_genes = []
        genes = set(genes)
        for gene in self.ae_input_genes:
            if gene in genes:
                overlapping_genes.append(gene)
        return overlapping_genes





if __name__ == '__main__':
    #def __init__(self,embedding_path,data_dir,knowledge_path,overlap_genes_path,ae_input_genes,tf_list_path):
    #embedding_path = '/nobackup/users/schaferd/ae_project_outputs/vanilla/get_model_info_epochs3_batchsize128_edepth2_ddepth2_lr0.0001_6-23_13.31.0/model_encoder_fold0.pth'
    embedding_path = '/nobackup/users/schaferd/ae_project_outputs/for_loop/moa_tests_epochs100_batchsize256_edepth2_ddepth4_lr0.0001_moa0.1_7-19_11.8.7/model_encoder_fold0.pth'
    data_dir = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/contrasts/'
    knowledge_path = '/nobackup/users/schaferd/ae_project_data/dorothea_tf_gene_relationship_knowledge/dorotheaSelectionA.tsv'
    overlap_genes_path = '/nobackup/users/schaferd/ko_eval_data/ae_data/overlap_list.pkl'
    ae_input_genes = '/nobackup/users/schaferd/ko_eval_data/ae_data/input_genes.pkl'
    tf_list_path = '/nobackup/users/schaferd/ko_eval_data/ae_data/embedding_tf_names.pkl'
    obj = ActivityInput(embedding_path,data_dir,knowledge_path,overlap_genes_path,ae_input_genes,tf_list_path)

