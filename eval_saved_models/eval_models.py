from eval import EvalModel
import os
import sys
from argparse import RawTextHelpFormatter
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class EvalModels():

    def __init__(self, param_dict):
        self.model_files = []
        for m in param_dict['model_dirs']:
            self.model_files = self.model_files + self.get_model_files(m)
        self.models = []

        for m in self.model_files:
            param_dict['model_path'] = m
            model = EvalModel(param_dict)
            self.models.append(model)

    def get_model_files(self, path):
        model_dirs = [path+'/'+f for f in os.listdir(path) if "fold" in f and "cycle" in f and os.path.isfile(path+f) == False]
        model_files = []
        for i, f in enumerate(model_dirs):
            paths = os.listdir(f)
            for p in paths:
                if ".pth" in p:
                    model_files.append(f+'/'+p)
        return model_files

    def create_rank_freq_heatmaps(self, treated_data_path, control_data_path):
        ranks, tfs = self.get_differential_activities(treated_data_path,control_data_path)
        freq_matrix, freq_df = self.create_freq_buckets(ranks,tfs,5)
        fig,ax = plt.subplots()
        fig.set_figwidth(20)
        fig.set_figheight(20)
        sns.heatmap(freq_df,ax=ax,cmap='Blues')
        plt.savefig('outputs/heatmap.png')

    def run_ko_tests(self,ko_data_path):
        auc_dict = {}
        for i,m in enumerate(self.models):
            fold = self.model_files[i].split('/')[-2].split('_')[0][-1]
            cycle = self.model_files[i].split('/')[-2][-1]
            new_path = os.getcwd()+'/outputs/'+'/'.join(self.model_files[i].split('/')[-4:-1])+'/ko_activities_cycle'+cycle+'_fold'+fold+'/'
            model_type = self.model_files[i].split('/')[-4]
            print(new_path)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            auc = m.run_ko_tests(ko_data_path,new_path)
            if model_type not in auc_dict:
                auc_dict[model_type] = [auc]
            else:
                auc_dict[model_type].append(auc)

        for key in auc_dict:
            path = os.getcwd()+'/outputs/'+key+'/knocktf_aucs.pkl'
            pd.to_pickle(auc_dict[key],path)

    def run_pert_tests(self,ko_data_path):
        auc_dict = {}
        for i,m in enumerate(self.models):
            fold = self.model_files[i].split('/')[-2].split('_')[0][-1]
            cycle = self.model_files[i].split('/')[-2][-1]
            print("fold",fold)
            print("cycle",cycle)
            new_path = os.getcwd()+'/outputs/'+'/'.join(self.model_files[i].split('/')[-4:-1])+'/ko_activities_cycle'+cycle+'_fold'+fold+'/'
            model_type = self.model_files[i].split('/')[-4]
            print(new_path)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            auc = m.run_pert_tests(ko_data_path,new_path)
            if model_type not in auc_dict:
                auc_dict[model_type] = [auc]
            else:
                auc_dict[model_type].append(auc)

        for key in auc_dict:
            path = os.getcwd()+'/outputs/'+key+'/aucs.pkl'
            pd.to_pickle(auc_dict[key],path)

        
    
    def get_differential_activities(self, treated_data_path, control_data_path):
        treated_activities = [m.pred_TF_activities(treated_data_path)[1] for m in self.models]
        control_activities = [m.pred_TF_activities(control_data_path)[1] for m in self.models]
        same_model_buckets = np.array([])
        tfs = list(treated_activities[0][0]['TFs'])
        print("tfs")
        print(tfs)
        print(len(control_activities))
        for m_runs in range(len(treated_activities)):
            m_ranks = np.array([])
            for sample in range(len(treated_activities[m_runs])):
                diff = (np.array(treated_activities[m_runs][sample]['activities'])-np.array(control_activities[m_runs][sample]['activities']))
                order = diff.argsort()
                rank = order.argsort()
                if m_ranks.shape[0] == 0:
                    m_ranks = np.array([rank])
                else:
                    m_ranks = np.vstack((m_ranks,rank))
            if same_model_buckets.shape[0] == 0:
                same_model_buckets = np.array([m_ranks])
            else:
                same_model_buckets = np.vstack((same_model_buckets,[m_ranks]))
        same_model_average = np.mean(same_model_buckets,axis=1)
        print("same model average")
        print(same_model_average)
        rank_df = pd.concat([pd.DataFrame({'m'+str(i):ranks},index=tfs) for i,ranks in enumerate(same_model_average)],axis=1)
        print(rank_df)
        rank_df.to_pickle('outputs/rank_df.pkl')
        return same_model_average, tfs

    def create_freq_buckets(self, rankings, tfs, bucket_size):
        div = int(float(np.max(rankings)) // bucket_size)
        print(div)
        freq_matrix = np.zeros((len(tfs),(div+1)))
        print(freq_matrix.shape)
        for model in rankings:
            for tf,rank in enumerate(model):
                freq_matrix[tf, int(rank // bucket_size)] += 1
        print(freq_matrix)
        df = pd.DataFrame(freq_matrix, index = tfs, columns = np.arange(0,(div+1)*bucket_size,bucket_size))
        return freq_matrix, df

        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to load and evaluate an autoencoder',formatter_class=RawTextHelpFormatter)
    parser.add_argument('--encoder_depth',type=int,required=False,default=2,help="number of hidden layers in encoder module (only applicable to FC and TF encoders)")
    parser.add_argument('--decoder_depth',type=int,required=False,default=2,help="number of hidden layers in decoder module (only applicable to FC and G decoders)")
    parser.add_argument('--train_data',type=str,required=True,help='Path to data used to train the network')
    parser.add_argument('--width_multiplier',type=int,required=False,default=1,help='multiplicative factor that determines width of hidden layers (only applies to FC, TF and G modules)')
    parser.add_argument('--relationships_filter',type=int,required=True,help='Minimum number of genes each TF must have relationships with in the prior knowledge')
    parser.add_argument('--prior_knowledge',type=str,required=True,help='Path to prior knowledge')

    args = parser.parse_args()

    params = {
        "model_dirs":['/nobackup/users/schaferd/ae_project_outputs/final_eval/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/','/nobackup/users/schaferd/ae_project_outputs/final_eval/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.56/'],
        #"model_dirs":['/nobackup/users/schaferd/ae_project_outputs/final_eval/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/','/nobackup/users/schaferd/ae_project_outputs/final_eval/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_14.26.53/'],
        "encoder_depth":args.encoder_depth,
        "decoder_depth":args.decoder_depth,
        "train_data":args.train_data,
        "width_multiplier":args.width_multiplier,
        "relationships_filter":args.relationships_filter,
        "prior_knowledge":args.prior_knowledge
    }

    obj = EvalModels(params)
    #obj.create_rank_freq_heatmaps('/nobackup/users/schaferd/drug_perturb_data/belinostat_dexamethasone_A549/untreated/samples.pkl','/nobackup/users/schaferd/drug_perturb_data/belinostat_dexamethasone_A549/belinostat_treated/samples.pkl')
    obj.run_ko_tests("/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/pos_neg_samples/")
    obj.run_pert_tests("/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/pos_neg_samples/")
    
