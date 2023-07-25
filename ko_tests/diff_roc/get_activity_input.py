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
        self.overlap_genes = overlap_genes
        self.tf_list = tf_list
        self.out_dir = out_dir
        self.ae_input_genes = ae_input_genes
        self.knowledge = knowledge

        pos_file = data_dir+'/pos_df.csv'
        neg_file = data_dir+'/neg_df.csv'

        self.positive_samples_df = pd.read_csv(pos_file,index_col=0)
        #self.positive_samples_df = pd.read_csv(pos_file)
        #self.index_to_ko_tfs = dict(self.positive_samples_df['Unnamed: 0'])
        self.index_to_ko_tfs = {id_ : id_.split('_')[1] for id_ in self.positive_samples_df.index}
        #self.positive_samples_df = self.positive_samples_df.drop(columns=['Unnamed: 0'])
        self.negative_samples_df = pd.read_csv(neg_file,index_col=0)
        #self.negative_samples_df = pd.read_csv(neg_file)
        print(self.negative_samples_df)
        #self.negative_samples_df = self.negative_samples_df.drop(columns=['Unnamed: 0'])
        
        self.pos_samples = self.filter_matrix(self.positive_samples_df)
        self.neg_samples = self.filter_matrix(self.negative_samples_df)

        #for i, pos_sample in enumerate(self.positive_samples):
        #    self.get_activities(pos_sample,self.negative_samples[i])

    def filter_matrix(self,df):
        temp_df = pd.DataFrame(columns = self.ae_input_genes)
        df0 = pd.DataFrame(0,index=[0],columns=temp_df.columns)

        input_df = pd.concat([df0,df],ignore_index=True)

        input_df = input_df.loc[1:,self.ae_input_genes]
        input_df = input_df.fillna(0)
        matrix = torch.from_numpy(np.array(input_df).astype(np.float)).to(device).float()
        print('ae input genes',len(self.ae_input_genes))
        return matrix


    def get_activities(self):
        pos_embedding = self.encoder(self.pos_samples).cpu().detach().numpy()
        neg_embedding = self.encoder(self.neg_samples).cpu().detach().numpy()
        diff_embedding = pos_embedding-neg_embedding
        #pert_tfs = self.positive_samples_df.index

        print('diff embedding',diff_embedding)
        #print('pert tfs',len(pert_tfs))
        print('tf list',len(self.tf_list))
        #diff_df = pd.DataFrame(data=diff_embedding,index=pert_tfs,columns=self.tf_list)
        diff_df = pd.DataFrame(data=diff_embedding,columns=self.tf_list,index=self.positive_samples_df.index)
        print('diff df',diff_df)
        diff_df.to_csv(self.out_dir+'/diff_activities.csv')
        with open(self.out_dir+'/ko_tf_index.pkl','wb+') as f:
            pkl.dump(self.index_to_ko_tfs,f)


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

