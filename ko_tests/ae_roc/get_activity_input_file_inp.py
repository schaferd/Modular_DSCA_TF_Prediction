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

model_class_path = os.environ["model_class_path"]
sys.path.insert(1,model_class_path)
from ae_encoder import AEEncoder

is_gpu = False
device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
    is_gpu = True
print("is gpu 1"+str(is_gpu))


class ActivityInput():
    def __init__(self,embedding_path,data_dir,knowledge_path,overlap_genes_path,ae_input_genes_path,tf_list_path):
        self.device = torch.device('cuda')
        if torch.cuda.is_available():
            is_gpu = True
        print("is gpu 1"+str(is_gpu))
        self.data_dir = data_dir
        self.embedding_path = embedding_path
        self.knowledge_path = knowledge_path
        self.model = torch.load(embedding_path).to(self.device)
        self.model.eval()
        self.data_files = [data_dir+f for f in os.listdir(data_dir) if os.path.isfile(data_dir+f) and 'signature' in f]

        self.embedding_dir = '/'.join(self.embedding_path.split('/')[:-1])
        self.save_path = '/'.join(self.embedding_path.split('/')[:-1])+'/pred_ko_activities/'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.overlap_list = self.open_pkl(overlap_genes_path)
        self.overlap_set = set(self.overlap_list)
        self.knowledge = self.get_knowledge()
        self.ae_input_genes = self.open_pkl(ae_input_genes_path)
        try:
            self.tf_list = self.open_pkl(self.embedding_dir+'/tfs_in_embedding.pkl')
        except:
            self.tf_list = self.open_pkl(tf_list_path)
        print("num data files", len(self.data_files))
        for data_file in self.data_files:
            act_start = time.time()
            self.activities = self.get_activities(data_file)
            print("activity time",(time.time()-act_start))
            break


    def open_pkl(self,pkl_file):
        with open(pkl_file,'rb') as f:
                return pkl.load(f)

    def get_activities(self,data_path):
        sample = self.get_ko_data(data_path)
        size = self.get_size(sample)
        sample = np.array(sample).astype(np.float)
        sample = torch.tensor(sample).float().to(self.device)
        embedding = self.model(sample).cpu().detach().numpy()
        split_data_path = data_path.split('/')[-1].split('.')
        print(split_data_path)
        csv_file = split_data_path[0]+'.'+split_data_path[1]+'.pred_activities.csv'
        csv_path = self.save_path+'/'+csv_file

        print("csv path")
        print(csv_path)

        data_struct = {'regulon':self.tf_list,'activities':embedding[0],'size':size}
        
        df = pd.DataFrame(data_struct)
        df.to_csv(csv_path)

    def get_ko_data(self,data_path):
        data = self.get_list_from_file(data_path)
        gene_expression_dict = {}
        gene_list = []
        for i,row in enumerate(data):
            temp = row.split(' ')
            genes = self.convert_gene_name_to_ensembl(temp[0].replace('"',''))
            if genes[0] is not None: 
                gene_list.extend(genes)
                gene_expression_dict[genes[0]] = float(temp[-1].strip())
        
        df = pd.DataFrame(gene_expression_dict,index=[0])
        print("genes in data",len(gene_list))

        overlapping_genes = self.get_overlapping_genes(df.columns)
        print("ae data overlap genes",len(overlapping_genes))
        ordered_exp = [gene_expression_dict[gene] for gene in overlapping_genes]
        #ordered_exp = stats.zscore(ordered_exp)
        

        #input_df = pd.DataFrame(0,index=[0],columns = self.ae_input_genes)
        input_df = pd.DataFrame(columns = self.ae_input_genes)
        #input_df.loc[len(df)] = 0
        #input_df = input_df.append(pd.Series(0, index=input_df.columns), ignore_index=True)
        #input_df.loc[len(input_df.index)] = pd.Series(0, index=input_df.columns)
        df0 = pd.DataFrame(0,index=[len(input_df.index)],columns=input_df.columns)
        input_df = pd.concat([input_df,df0], ignore_index=True)
        input_df.loc[0,overlapping_genes] = ordered_exp 
    
        return input_df

    def get_overlapping_genes(self,genes):
        overlapping_genes = []
        genes = set(genes)
        print("ae genes", len(self.ae_input_genes))
        for gene in self.ae_input_genes:
            if gene in genes:
                overlapping_genes.append(gene)
        return overlapping_genes
            

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

    def get_knowledge(self):
        df = pd.read_csv(self.knowledge_path,sep='\t',low_memory=False)
        tf_gene_dict = {}
        for i,row in df.iterrows():
            translated_target = self.convert_gene_name_to_ensembl(row['target'])[0]
            if translated_target is not None:
                if row['tf'] in tf_gene_dict and translated_target in self.overlap_set:
                    tf_gene_dict[row['tf']].append(row['target'])
                elif translated_target in self.overlap_set:
                    tf_gene_dict[row['tf']] = [row['target']]
        return tf_gene_dict






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

