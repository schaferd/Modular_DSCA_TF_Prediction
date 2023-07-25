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

class getROCCurve():
    def __init__(self,diff_activities,id_to_tf):
        self.id_to_tf = id_to_tf
        self.diff_activities = diff_activities
        self.scaled_rankings = self.rank_matrix(self.diff_activities)
        self.perturbation_df = self.get_perturbation_info()
        self.tfs_of_interest = self.get_tfs_of_interest()
        self.auc = self.get_roc()




    def rank_matrix(self,df):
        #ranked_matrix = self.aggregate_activities.rank(axis = 1,method='min',na_option='keep',ascending='False')
        #scaled_rank_matrix = ranked_matrix/ranked_matrix.max(axis=0)
        return df.rank(axis=1)/df.shape[0]

    def get_perturbation_info(self):
        tfs = self.scaled_rankings.columns
        self.scaled_rankings = self.scaled_rankings.reset_index(names='Sample_ID')
        rank_df = pd.melt(self.scaled_rankings,id_vars=['Sample_ID'],value_vars=tfs,value_name='scaled ranking',var_name='TF',ignore_index=False)

        #rank_df = pd.melt(self.scaled_rankings,value_vars=self.scaled_rankings.columns,ignore_index=False)
        #rank_df.rename(columns={'Unnamed: 0':'Sample'},inplace=True)
        #print(rank_df)
        #rank_df.rename({'value':'scaled ranking'},axis=1,inplace=True)
        #activity_df = pd.melt(self.aggregate_activities,value_vars=self.scaled_rankings.columns,ignore_index=False)
        #activity_df.rename({'value':'pred activity'},axis=1,inplace=True)
        #rank_df['pred activity'] = activity_df['pred activity']
        per_list = [self.id_to_tf[sample] for sample in rank_df['Sample_ID'].tolist()]
        #per_list = [name.split('.')[0] for name in rank_df['Sample'].tolist()]
        rank_df['perturbed tf'] = per_list
        return rank_df

    def get_tfs_of_interest(self):
        df_tf_of_interest = self.perturbation_df.copy()
        #df_tf_of_interest.reset_index(inplace=True)
        #df_tf_of_interest.rename(columns={'index':'regulon'},inplace=True)
        pert_tfs = set(df_tf_of_interest['perturbed tf'].tolist())
        pred_tfs = set(df_tf_of_interest['TF'].tolist())
        #df_tf_of_interest['tf'] = df_tf_of_interest.index
        tfs_of_interest = list(pert_tfs.intersection(pred_tfs))
        df_tf_of_interest = df_tf_of_interest[df_tf_of_interest['TF'].isin(tfs_of_interest)]
        df_tf_of_interest['is tf perturbed'] = (df_tf_of_interest['TF'] == df_tf_of_interest['perturbed tf'])
        #df_tf_of_interest.fillna(0,inplace=True)
        df_tf_of_interest.dropna(inplace=True)
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


def load_activity_file(activity_file,exp_id):
    df = pd.read_csv(activity_file)
    df.index = [exp_id]
    return df

def aggregate_matrix(activities):
    df = pd.concat(activities.values(),axis=0)
    #df = pd.concat(activities_list,ignore_index=True).set_index('Unnamed: 0',drop=True).T
    #df = pd.DataFrame(activities).rename(columns=self.exp_ids).T
    return df

def knocktf_diff_activities(activity_dir,method):
    activity_dir = activity_dir
    activity_files = [f for f in os.listdir(activity_dir) if (os.path.isfile('/'.join([activity_dir,f])) and method in f) and "diff_activities" not in f]

    exp_ids = {f:'_'.join((f.split('.')[0]).split('_')[2:5]) for f in activity_files }

    tf = {f: f.split('.')[1] for f in activity_files}

    id_to_tf = {exp_ids[f]:tf[f] for f in activity_files}
    activities = {f:load_activity_file('/'.join([activity_dir,f]),exp_ids[f])  for f in activity_files}

    control_activities = aggregate_matrix({f:activities[f] for f in activities.keys() if 'control' in f})
    treated_activities = aggregate_matrix({f:activities[f] for f in activities.keys() if 'treated' in f}).loc[control_activities.index,:]

    diff_activities = control_activities - treated_activities
    diff_activities.to_csv(activity_dir+'/'+method+'diff_activities.csv',sep='\t')
    return diff_activities, id_to_tf
        

def dorothea_benchmark_diff_activities(activity_dir,method):
    activity_files = [f for f in os.listdir(activity_dir) if (os.path.isfile('/'.join([activity_dir,f])) and method in f) and "diff_activities" not in f]

    exp_ids = {f:'_'.join((f.split('.')[0]).split('_')[2:4]) for f in activity_files }

    tf = {f: f.split('.')[0].split('_')[2] for f in activity_files}

    id_to_tf = {exp_ids[f]:tf[f] for f in activity_files}
    activities = {f:load_activity_file('/'.join([activity_dir,f]),exp_ids[f])  for f in activity_files}

    positive_activities = aggregate_matrix({f:activities[f] for f in activities.keys() if 'positive' in f})
    negative_activities = aggregate_matrix({f:activities[f] for f in activities.keys() if 'negative' in f}).loc[positive_activities.index,:]

    diff_activities = positive_activities - negative_activities
    diff_activities.to_csv(activity_dir+'/'+method+'diff_activities.csv',sep='\t')
    return diff_activities, id_to_tf

if __name__ == '__main__':
    #activity_dir = '/nobackup/users/schaferd/ae_project_data/ko_data/TF_activities_dorothea_relconn10/'
    #activity_dir = '/nobackup/users/schaferd/ae_project_data/encode_ko_data/dorothea_activities/'

    knocktf_activity_dir = '/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/viper_data/TF_activities'
    dorothea_benchmark_activity_dir = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/diff_TF_activities'

    print("KNOCKTF VIPER")
    knocktf_diff_activities_viper, knocktf_id_to_tf_viper = knocktf_diff_activities(knocktf_activity_dir,'VIPER')
    knocktf_VIPER_obj = getROCCurve(knocktf_diff_activities_viper, knocktf_id_to_tf_viper)

    print("KNOCKTF SCENIC")
    knocktf_diff_activities_SCENIC, knocktf_id_to_tf_SCENIC = knocktf_diff_activities(knocktf_activity_dir,'SCENIC')
    knocktf_SCENIC_obj = getROCCurve(knocktf_diff_activities_SCENIC, knocktf_id_to_tf_SCENIC)

    print("dorothea VIPER")
    dorotheaB_diff_activities_viper,dB_id_to_tf_viper = dorothea_benchmark_diff_activities(dorothea_benchmark_activity_dir,'VIPER')
    dorotheaB_VIPER_obj = getROCCurve(dorotheaB_diff_activities_viper,dB_id_to_tf_viper)

    print("dorothea SCENIC")
    dorotheaB_diff_activities_SCENIC, dB_id_to_tf_SCENIC = dorothea_benchmark_diff_activities(dorothea_benchmark_activity_dir,'SCENIC')
    dorotheaB_SCENIC_obj = getROCCurve(dorotheaB_diff_activities_SCENIC,dB_id_to_tf_SCENIC)

    #activity_dir = '/nobackup/users/schaferd/ae_project_data/ko_data/sample_tf_activities/'
    #decoupleR_obj = getROCCurve(activity_dir,'decoupleR')
    #SCENIC_obj = getROCCurve(activity_dir,'SCENIC')

    #print("viper",VIPER_obj.auc)
    #print("decoupleR",decoupleR_obj.auc)
    """
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
    """
