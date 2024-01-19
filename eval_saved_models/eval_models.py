from eval import EvalModel
from eval_state_dict import EvalStateDict
import os
import sys
from argparse import RawTextHelpFormatter
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from get_roc_curve import getROC
from get_pert_roc_curves import getPertROC

class EvalModels():

    def __init__(self, param_dict):
        self.model_files = []
        self.param_dict = param_dict
        for m in param_dict['model_dirs']:
            self.model_files = self.model_files + self.get_model_files(m)
        self.models = []
        self.encoder = None
        self.decoder = None

        if os.environ['SHALLOW_ENCODER'] == "true":
            self.encoder = 'shallow'
        elif os.environ['TF_GROUPED_FC_INDEP_ENCODER'] == "true":
            self.encoder = 'tf'
        elif os.environ['FULLY_CONNECTED_ENCODER'] == "true":
            self.encoder = 'fc'
        else:
            raise ValueError()

        if os.environ['SHALLOW_DECODER'] == "true":
            self.decoder = 'shallow'
        elif os.environ['GENE_GROUPED_FC_INDEP_DECODER'] == "true":
            self.decoder = 'g'
        elif os.environ['FULLY_CONNECTED_DECODER'] == "true":
            self.decoder = 'fc'
        else:
            raise ValueError()

        self.out_dir = 'outputs/'+self.encoder+'_'+self.decoder+'/'
        print(self.out_dir)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

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
        plt.savefig(self.out_dir+'heatmap.png')

    def create_performance_vs_ko_rank_curve(self,ko_data_path):
        #cutoffs = [0.05,0.1,0.2,0.3,0.4,0.5]
        cutoffs = [0.1,0.2,0.3,0.4,0.5]
        avg_aucs = []
        auc_dict = {}
        for c in cutoffs:
            avg_auc, aucs = self.run_ko_tests(ko_data_path,control='filtered_control_'+str(c)+'.csv',treated='filtered_treated_'+str(c)+'.csv')
            auc_dict[c] = aucs
            avg_aucs.append(avg_auc)

        pd.to_pickle(avg_aucs,self.out_dir+'rank_vs_auc_avg_aucs.pkl')
        pd.to_pickle(cutoffs,self.out_dir+'rank_vs_auc_cutoffs.pkl')
        pd.to_pickle(auc_dict,self.out_dir+'rank_vs_auc_auc_dict.pkl')

        plt.clf()
        plt.plot(cutoffs,avg_aucs)
        plt.ylabel("Avg ROC AUC")
        plt.xlabel("Treated Rank Cutoffs")
        plt.title(self.encoder+' '+self.decoder)
        plt.savefig(self.out_dir+"auc_vs_ko_rank.png")

    def run_ko_tests(self,ko_data_path,control="control.csv",treated="treated.csv",attribution=False,recon_corr=False):
        auc_dict = {}
        aucs = []
        average_activities = None
        df_counter = 0
        koed_tfs = None
        for i,m in enumerate(self.models):
            fold = self.model_files[i].split('/')[-2].split('_')[0][-1]
            cycle = self.model_files[i].split('/')[-2][-1]
            new_path = os.getcwd()+'/'+self.out_dir+'/'.join(self.model_files[i].split('/')[-4:-1])+'/ko_activities_cycle'+cycle+'_fold'+fold+'/'
            model_type = self.model_files[i].split('/')[-4]

            if not os.path.exists(new_path):
                os.makedirs(new_path)
            auc, activity_df, koed_tfs = m.run_ko_tests(ko_data_path,new_path,control=control,treated=treated)

            if average_activities is None:
                average_activities = activity_df
            else:
                average_activities = average_activities.add(activity_df)
            df_counter += 1

            aucs.append(auc)

            if model_type not in auc_dict:
                auc_dict[model_type] = [auc]
            else:
                auc_dict[model_type].append(auc)
            
            if attribution:
                control_attr = m.get_attribution(ko_data_path+control,new_path,pickle=False)
                treated_attr = m.get_attribution(ko_data_path+treated,new_path,pickle=False)
            
            if recon_corr:
                pd.to_pickle(m.get_reconstruction(ko_data_path+control,pickle=False)[-1],new_path+'knocktf_recon_corr_control.pkl')
                pd.to_pickle(m.get_reconstruction(ko_data_path+treated,pickle=False)[-1],new_path+'knocktf_recon_corr_treated.pkl')


        average_activities = average_activities / df_counter
        average_activities.to_csv(self.out_dir+'ensemble_activities.csv',sep='\t')

        print("getting ROC")
        ko_obj = getROC(average_activities, koed_tfs, 'ko_roc.png')
        print("ko auc",ko_obj.auc)

        for key in auc_dict:
            path = os.getcwd()+'/'+self.out_dir+key+'/knocktf_aucs.pkl'
            pd.to_pickle(auc_dict[key],path)

        return ko_obj.auc, aucs

    def run_pert_tests(self,ko_data_path,recon_corr=False):
        auc_dict = {}
        average_activities = None
        df_counter = 0
        koed_tfs = None
        for i,m in enumerate(self.models):
            fold = self.model_files[i].split('/')[-2].split('_')[0][-1]
            cycle = self.model_files[i].split('/')[-2][-1]
            new_path = os.getcwd()+'/'+self.out_dir+'/'.join(self.model_files[i].split('/')[-4:-1])+'/ko_activities_cycle'+cycle+'_fold'+fold+'/'
            model_type = self.model_files[i].split('/')[-4]

            if not os.path.exists(new_path):
                os.makedirs(new_path)
            auc, activity_df, index_to_koed_tfs = m.run_pert_tests(ko_data_path,new_path)
            pd.to_pickle(index_to_koed_tfs,new_path+'/pert_index_to_kotf.pkl')

            if average_activities is None:
                average_activities = activity_df
            else:
                average_activities = average_activities.add(activity_df)
            df_counter += 1

            if model_type not in auc_dict:
                auc_dict[model_type] = [auc]
            else:
                auc_dict[model_type].append(auc)

            if recon_corr:
                pd.to_pickle(m.get_reconstruction(ko_data_path+'pos_df.csv',pickle=False,csv=True)[-1],new_path+'dorothea_recon_corr_control.pkl')
                pd.to_pickle(m.get_reconstruction(ko_data_path+'neg_df.csv',pickle=False,csv=True)[-1],new_path+'dorothea_recon_corr_treated.pkl')
        average_activities = average_activities / df_counter
        average_activities.to_csv(self.out_dir+'dorothea_ensemble_activities.csv',sep='\t')
        print("average_activities")
        print(average_activities)

        ko_obj = getPertROC(average_activities, index_to_koed_tfs,'pert_roc.png')
        print("pert auc",ko_obj.auc)

        for key in auc_dict:
            path = os.getcwd()+'/'+self.out_dir+key+'/aucs.pkl'
            pd.to_pickle(auc_dict[key],path)
        
        return ko_obj.auc

    def get_embeddings(self, data_path):
        activities = []
        for i,m in enumerate(self.models):
            output, output_dfs = m.pred_TF_activities(data_path)
            activities.append(output)

        out_file = os.getcwd()+'/'+self.out_dir+'/'+data_path.split('/')[-1]+'.activities.pkl'
        pd.to_pickle(activities,out_file)
        return activities

    def get_avg_missing_expression(self,input_data_path):
        new_input_dfs = []
        for m in self.models:
            new_input_data = m.infer_missing_gene_expression(input_data_path=input_data_path)
            new_input_dfs.append(new_input_data)

        new_input_np = np.vstack([[df.to_numpy()] for df in new_input_dfs])
        avg_input = pd.DataFrame(new_input_np.mean(axis=0),columns=new_input_dfs[0].columns,index=new_input_dfs[0].index)
        return avg_input

        
    
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
        print("rank df")
        print(rank_df)
        rank_df.to_pickle(self.out_dir+'rank_df.pkl')
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

    def get_consensus_model(self):
        state_dicts = [m.model.state_dict() for m in self.models]
        avg_dict = {}
        print("state_dicts")
        print(state_dicts)

        for key in state_dicts[0]:
            print("dict 0")
            print(state_dicts[0][key])
            key_sum = None
            for m in state_dicts:
                if key_sum is None:
                    key_sum = m[key]
                else:
                    key_sum += m[key]
            avg_dict[key] = key_sum/len(state_dicts)
            print("average dict")
            print(avg_dict[key])

        params = self.param_dict
        params['model'] = avg_dict
        avg_model = EvalStateDict(params)
        return avg_model


        



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
        #FINAL EVAL FC-G
        #"model_dirs":['/nobackup/users/schaferd/ae_project_outputs/final_eval/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/','/nobackup/users/schaferd/ae_project_outputs/final_eval/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.56/'],
        #FINAL EVAL S-S
        #"model_dirs":['/nobackup/users/schaferd/ae_project_outputs/final_eval/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/','/nobackup/users/schaferd/ae_project_outputs/final_eval/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_14.26.53/'],


        #MODEL SEARCH 
        #FC-FC
        #"model_dirs":['/nobackup/users/schaferd/ae_project_outputs/model_eval/save_model_fc-fc_epochs100_batchsize128_enlr0.0001_delr0.0001_moa1.0_rel_conn10_5-31_17.51.41/'],
        #FC-G
        #"model_dirs":['/nobackup/users/schaferd/ae_project_outputs/model_eval/save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_rel_conn10_5-31_17.52.2/'],
        #FC-S
        #"model_dirs":['/nobackup/users/schaferd/ae_project_outputs/model_eval/save_model_fc-shallow_epochs100_batchsize128_enlr0.0001_delr0.001_moa1.0_rel_conn10_5-31_17.52.35/'],

        #S-FC
        #"model_dirs":['/nobackup/users/schaferd/ae_project_outputs/model_eval/save_model_shallow-fc_epochs100_batchsize128_enlr0.001_delr0.0001_moa1.0_rel_conn10_6-1_10.42.8/'],
        #S-G
        #"model_dirs":['/nobackup/users/schaferd/ae_project_outputs/model_eval/save_model_shallow-genefc_epochs100_batchsize128_enlr0.01_delr0.01_moa1.0_rel_conn10_6-1_10.42.44/'],
        #S-S
        #"model_dirs":['/nobackup/users/schaferd/ae_project_outputs/model_eval/save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_moa1.0_rel_conn10_6-1_10.43.21/'],

        #T-FC
        #"model_dirs":['/nobackup/users/schaferd/ae_project_outputs/model_eval/save_model_tffc-fc_epochs100_batchsize128_enlr0.001_delr0.0001_moa1.0_rel_conn10_5-31_17.54.10/'],
        #T-G
        #"model_dirs":['/nobackup/users/schaferd/ae_project_outputs/model_eval/save_model_tffc-genefc_epochs100_batchsize128_enlr0.001_delr0.01_moa1.0_rel_conn10_5-31_23.34.32/'],
        #T-S
        #"model_dirs":['/nobackup/users/schaferd/ae_project_outputs/model_eval/save_model_tffc-shallow_epochs100_batchsize128_enlr0.001_delr0.01_moa1.0_rel_conn10_6-1_10.41.11/'],


        #MOA TEST
        #FC-G
        #"model_dirs":['/nobackup/users/schaferd/ae_project_outputs/moa_tests/saved_moa_test_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_rel_conn10_7-31_10.22.55/','/nobackup/users/schaferd/ae_project_outputs/moa_tests/saved_moa_test_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_rel_conn10_7-31_10.23.3/'],
        "model_dirs":['/nobackup/users/schaferd/ae_project_outputs/moa_tests/saved_no_moa_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_rel_conn10_11-15_12.4.41/','/nobackup/users/schaferd/ae_project_outputs/moa_tests/saved_moa_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_moa1.0_rel_conn10_11-15_10.54.12/'],



        "encoder_depth":args.encoder_depth,
        "decoder_depth":args.decoder_depth,
        "train_data":args.train_data,
        "width_multiplier":args.width_multiplier,
        "relationships_filter":args.relationships_filter,
        "prior_knowledge":args.prior_knowledge
    }

    obj = EvalModels(params)
    #obj.create_rank_freq_heatmaps('/nobackup/users/schaferd/drug_perturb_data/belinostat_dexamethasone_A549/untreated/samples.pkl','/nobackup/users/schaferd/drug_perturb_data/belinostat_dexamethasone_A549/belinostat_treated/samples.pkl')
    #obj.create_performance_vs_ko_rank_curve("/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/fc_filtered/")
    #obj.get_embeddings('/nobackup/users/schaferd/ae_project_data/hdf_gene_expression_data/agg_data.pkl')
    #obj.get_embeddings('/nobackup/users/schaferd/ae_project_data/encode_ko_data/combined_processed_df.pkl')
    #obj.run_ko_tests("/nobackup/users/schaferd/ae_project_data/encode_ko_data/")

    #obj.models[0].infer_missing_gene_expression('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/relevant_pos_neg_samples/pos_df.pkl').to_csv('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/relevant_pos_neg_samples/pos_df.csv')
    #obj.get_avg_missing_expression('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/relevant_pos_neg_samples/pos_df.pkl').to_csv('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/pos_neg_samples/pos_df.csv')
    #obj.get_avg_missing_expression('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/relevant_pos_neg_samples/neg_df.pkl').to_csv('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/pos_neg_samples/neg_df.csv')
    #obj.models[0].gene_exp_imputation_quality('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/relevant_pos_neg_samples/neg_df.pkl',AE=True,KNN_init=True)
    #obj.models[0].gene_exp_imputation_quality('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/relevant_pos_neg_samples/neg_df.pkl',AE=True,KNN_init=False)
    #obj.models[0].gene_exp_imputation_quality('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/relevant_pos_neg_samples/neg_df.pkl',AE=False)

    #obj.models[0].KNN_missing_gene_imputation(input_data_path='/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/relevant_pos_neg_samples/neg_df_nan.pkl').to_csv('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/relevant_pos_neg_samples/neg_df.csv')
    #obj.models[0].KNN_missing_gene_imputation(input_data_path='/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/relevant_pos_neg_samples/pos_df_nan.pkl').to_csv('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/relevant_pos_neg_samples/pos_df.csv')

    #obj.models[0].KNN_missing_gene_imputation(input_data_path='/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/control_nan.pkl').to_csv('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/control_df.csv',sep='\t')
    #obj.models[0].KNN_missing_gene_imputation(input_data_path='/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/treated_nan.pkl').to_csv('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/treated_df.csv',sep='\t')
    #obj.run_ko_tests("/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/",'control_df.csv','treated_df.csv',attribution=False,recon_corr=True)

    obj.run_pert_tests("/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/relevant_pos_neg_samples/",recon_corr=True)
    #obj.run_pert_tests("/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/pos_neg_samples/")

    #avg_model = obj.get_consensus_model()
    #avg_model.run_ko_tests("/nobackup/users/schaferd/ae_project_data/ko_data/","outputs/")
    
