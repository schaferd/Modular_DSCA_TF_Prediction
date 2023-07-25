import argparse
import sparselinear
import gc
from argparse import RawTextHelpFormatter
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import subprocess
from torch.utils.checkpoint import checkpoint_sequential
import sys
import time
import random
import torch
from captum.attr import IntegratedGradients
import pandas as pd
from torch import nn
import torch
from collections import OrderedDict
import shutil
import pickle as pkl

sys.path.insert(1,os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from moa import MOA
from data_class import CellTypeDataset
from ae_model import AE
from eval_funcs import get_correlation,get_ko_roc_curve, get_knocktf_ko_roc_curve#, plot_ko_rank_vs_connection
from figures import plot_input_vs_output,create_test_vs_train_plot,create_corr_hist,create_moa_figs,TF_ko_heatmap

from data_processing import DataProcessing

device = torch.device('cpu')

is_gpu = False
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    is_gpu = True
print("is gpu "+str(is_gpu))

encoder_path = os.environ["encoder_path"]
print(encoder_path)
sys.path.insert(1,encoder_path)
print(encoder_path)
from encoder import AEEncoder

decoder_path = os.environ["decoder_path"]
sys.path.insert(1,decoder_path)
print(decoder_path)
from decoder import AEDecoder

class EvalModel():
    def __init__(self, param_dict):
        self.train_data = param_dict["train_data"]
        self.data_obj = DataProcessing(self.train_data,param_dict['prior_knowledge'],128,param_dict['relationships_filter'])
        self.encoder = AEEncoder(data=self.data_obj,dropout_rate=0,batch_norm=0,width_multiplier=param_dict['width_multiplier'],depth=param_dict['encoder_depth'])
        self.decoder = AEDecoder(data=self.data_obj,dropout_rate=0,batch_norm=0,width_multiplier=param_dict['width_multiplier'],depth=param_dict['decoder_depth'])
        self.model = AE(self.encoder,self.decoder).to(device)
        self.model.load_state_dict(torch.load(param_dict['model_path']))
        self.model.eval()

    def pred_TF_activities(self, input_data_path):
        input_data, input_tensor = self.load_input_data(input_data_path)
        output = self.model.encoder(input_tensor).cpu().detach().numpy()
        output_dfs = [pd.DataFrame({'TFs':self.data_obj.tfs,'activities':o}) for o in output]
        return output, output_dfs
        
    def get_reconstruction(self, input_data_path):
        input_data, input_tensor = self.load_input_data(input_data_path)
        output = self.model(input_tensor).cpu().detach().numpy()
        output_dfs = [pd.DataFrame({'Genes':self.data_obj.overlap_list,'counts':o}) for o in output]
        return output, output_dfs

    def infer_missing_gene_expression(self, input_data_path):
        """
        Input dataframe should have 
            - NaNs where gene values are missing
            - z-scored across genes using non-NaN values
            - should be a pickle file
        """

        #filter df to only input genes
        input_data = pd.read_pickle(input_data_path)
        temp_df = pd.DataFrame(columns=self.data_obj.input_genes)
        overlap_genes = list(set(self.data_obj.input_genes).intersection(set(input_data.columns)))
        input_data = input_data.loc[:,overlap_genes]
        input_data = pd.concat([pd.DataFrame(columns=self.data_obj.input_genes),input_data],axis=0)

        #create mask for missing genes (1 for missing gene, 0 otherwise)
        missing_gene_mask = input_data.notna()
        missing_gene_mask = missing_gene_mask.replace({True:0,False:1})

        input_data = input_data.fillna(0)
        input_tensor = torch.tensor(input_data.to_numpy()).float().to(device)

        #send data through AE and apply mask
        output = self.model(input_tensor).cpu().detach().numpy()
        output_df = pd.DataFrame(output,columns=self.data_obj.input_genes,index=input_data.index)
        missing_gene_values = missing_gene_mask*output_df
        missing_gene_values_np = missing_gene_values.to_numpy()
        print(missing_gene_values_np[np.nonzero(missing_gene_values_np)])

        #add output (with mask applied) to input data, return result
        new_input_data = input_data.add(missing_gene_values)

        return new_input_data

    def get_attribution(self, input_data_path, outpath, pickle=True):
        input_data, input_tensor = self.load_input_data(input_data_path,pickle=pickle)
        ig = IntegratedGradients(self.model.encoder)
        tf_attr_dict = {}
        for i,tf in enumerate(self.data_obj.tfs):
            attr,delta = ig.attribute(input_tensor,target=i,return_convergence_delta=True)
            attr = attr.detach().cpu().numpy()
            tf_attr_dict[tf] = attr
        
        pd.to_pickle(tf_attr_dict,outpath+'/tf_attr_dict.pkl')
        pd.to_pickle(self.data_obj.input_genes, outpath+'/input_genes.pkl')
        return tf_attr_dict

    def load_input_data(self, input_data_path,pickle=True):
        if pickle:
            input_data = pd.read_pickle(input_data_path)
        else:
            input_data = pd.read_csv(input_data_path,index_col=0,sep='\t')

        try:
            input_data = input_data.loc[:,self.data_obj.input_genes]
        except:
            overlap_genes = list(set(self.data_obj.input_genes).intersection(set(input_data.columns)))
            overlap_genes.sort()
            input_data = input_data.loc[:,overlap_genes]
            input_data = pd.concat([pd.DataFrame(columns=self.data_obj.input_genes),input_data],axis=0).astype(float)
        input_tensor = torch.tensor(input_data.to_numpy()).float().to(device)
        return input_data, input_tensor

    def run_pert_tests(self, ko_data_path,save_path):
        auc, activity_df, ranked_df, ko_tf_ranks, index_to_koed_tfs = get_ko_roc_curve(self.data_obj,ko_data_path,self.model.encoder,save_path)
        return auc, activity_df, index_to_koed_tfs

    def run_ko_tests(self, ko_data_path, save_path,control="control.csv",treated="treated.csv"):
        auc, activity_df, ranked_df, ko_tf_ranks, koed_tfs = get_knocktf_ko_roc_curve(self.data_obj,ko_data_path,self.model.encoder,save_path,control=control,treated=treated)
        print("activity df")
        print(activity_df)
        return auc, activity_df, koed_tfs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to load and evaluate an autoencoder',formatter_class=RawTextHelpFormatter)
    parser.add_argument('--model_path',type=str,required=True,help="Path to .pth state dict file")
    parser.add_argument('--encoder_depth',type=int,required=False,default=2,help="number of hidden layers in encoder module (only applicable to FC and TF encoders)")
    parser.add_argument('--decoder_depth',type=int,required=False,default=2,help="number of hidden layers in decoder module (only applicable to FC and G decoders)")
    parser.add_argument('--train_data',type=str,required=True,help='Path to data used to train the network')
    parser.add_argument('--width_multiplier',type=int,required=False,default=1,help='multiplicative factor that determines width of hidden layers (only applies to FC, TF and G modules)')
    parser.add_argument('--relationships_filter',type=int,required=True,help='Minimum number of genes each TF must have relationships with in the prior knowledge')
    parser.add_argument('--prior_knowledge',type=str,required=True,help='Path to prior knowledge')

    args = parser.parse_args()

    params = {
        "model_path":args.model_path,
        "encoder_depth":args.encoder_depth,
        "decoder_depth":args.decoder_depth,
        "train_data":args.train_data,
        "width_multiplier":args.width_multiplier,
        "relationships_filter":args.relationships_filter,
        "prior_knowledge":args.prior_knowledge
    }
    evaluation = EvalModel(params)
    print(evaluation.pred_TF_activities('/nobackup/users/schaferd/drug_perturb_data/belinostat_dexamethasone_A549/untreated/samples.pkl'))
    #evaluation.run_pert_tests("/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/pos_neg_samples/")
    #evaluation.run_ko_tests("/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/pos_neg_samples/")
