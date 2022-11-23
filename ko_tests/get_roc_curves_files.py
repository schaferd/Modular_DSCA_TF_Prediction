import numpy as np
import sys
import os
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
sys.path.insert(1,'./ae_roc')
import get_activity_input_file_inp as gai
import time

class getROCCurve():
    def __init__(self,ae_args={}):
        self.ae_args = ae_args 
        self.activity_files = None
        activity_dir = None
        if len(self.ae_args.keys()) > 0:
            act_inp_start = time.time()
            obj = gai.ActivityInput(ae_args['embedding_path'],ae_args['data_dir'],ae_args['knowledge_path'],ae_args['overlap_genes_path'],ae_args['ae_input_genes'],ae_args['tf_list_path'])
            print("activity input time:",(time.time()-act_inp_start))
            self.activity_files = {'.'.join(f.split('.')[:2]):f for f in os.listdir(obj.save_path) if os.path.isfile('/'.join([obj.save_path,f])) and 'pred_activities' in f}
            activity_dir = obj.save_path
        else:
            viper_activity_dir = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/contrasts/'
            self.activity_files = {'.'.join(f.split('.')[:2]):f for f in os.listdir(viper_activity_dir) if os.path.isfile('/'.join([viper_activity_dir,f])) and 'viper_pred.csv' in f}
            activity_dir = viper_activity_dir
        roc_time = time.time()
        self.activities = {f:self.load_activity_file(''.join([activity_dir,self.activity_files[f]]),f)  for f in self.activity_files}
        self.aggregate_activities = self.aggregate_matrix()
        self.scaled_rankings = self.rank_matrix()
        self.perturbation_df = self.get_perturbation_info()
        self.tfs_of_interest = self.get_tfs_of_interest()
        self.get_roc()
        print("roc time",(time.time()-roc_time))


    def load_activity_file(self,activity_file,exp_id):
        #returns pandas df
        df = pd.read_csv(activity_file)
        df['Sample'] = exp_id
        return df

    def aggregate_matrix(self):
        activities_list = list(self.activities.values())
        df = pd.concat(activities_list,ignore_index=True)
        samples = np.unique(df['Sample'].to_numpy()).tolist()
        df = pd.pivot_table(df,index=['regulon'],columns =['Sample'],values='activities') 
        df = df.dropna()
        return df

    def rank_matrix(self):
        ranked_matrix = self.aggregate_activities.rank(axis = 0,method='min',na_option='keep',ascending='False')
        scaled_rank_matrix = ranked_matrix/ranked_matrix.max(axis=0)
        return scaled_rank_matrix

    def get_perturbation_info(self):
        rank_df = pd.melt(self.scaled_rankings,value_vars=self.scaled_rankings.columns,ignore_index=False)
        rank_df.rename({'value':'scaled ranking'},axis=1,inplace=True)
        activity_df = pd.melt(self.aggregate_activities,value_vars=self.scaled_rankings.columns,ignore_index=False)
        activity_df.rename({'value':'pred activity'},axis=1,inplace=True)
        rank_df['pred activity'] = activity_df['pred activity']
        per_list = [name.split('.')[0] for name in rank_df['Sample'].tolist()]
        rank_df['perturbed tf'] = per_list
        return rank_df

    def get_tfs_of_interest(self):
        df_tf_of_interest = self.perturbation_df.copy()
        df_tf_of_interest.reset_index(inplace=True)
        pert_tfs = set(df_tf_of_interest['perturbed tf'].tolist())
        pred_tfs = set(df_tf_of_interest['regulon'].tolist())
        #df_tf_of_interest['tf'] = df_tf_of_interest.index
        tfs_of_interest = list(pert_tfs.intersection(pred_tfs))
        df_tf_of_interest = df_tf_of_interest[df_tf_of_interest['regulon'].isin(tfs_of_interest)]
        df_tf_of_interest['is tf perturbed'] = (df_tf_of_interest['regulon'] == df_tf_of_interest['perturbed tf'])
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
            plt.savefig(self.get_ae_save_path()+"/roc_ae.png")
        else:
            plt.savefig('roc_viper.png')

    def get_ae_save_path(self):
        embedding_path = '/'.join(self.ae_args['embedding_path'].split('/')[:-1])
        return embedding_path



        

if __name__ == '__main__':
    ae_args = {
        'embedding_path' : '/nobackup/users/schaferd/ae_project_outputs/for_loop/AB_tanh_batch_norm_test_epochs100_batchsize128_edepth2_ddepth4_lr0.001_moa1.0_do0.3_batchnorm_rel_conn3_8-12_12.35.46/model_encoder_fold0.pth',
        'data_dir' : '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/contrasts/',
        'knowledge_path' : '/nobackup/users/schaferd/ae_project_data/dorothea_tf_gene_relationship_knowledge/dorotheaSelectionAB.tsv',
        'overlap_genes_path' : '/nobackup/users/schaferd/ko_eval_data/ae_data/overlap_list.pkl',
        'ae_input_genes' : '/nobackup/users/schaferd/ko_eval_data/ae_data/input_genes.pkl',
        'tf_list_path' : '/nobackup/users/schaferd/ko_eval_data/ae_data/embedding_tf_names.pkl'
    }
    obj = getROCCurve(ae_args={})
    #obj = getROCCurve(ae_args=ae_args)
