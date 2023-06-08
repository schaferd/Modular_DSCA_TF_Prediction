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

"""
is_gpu = False
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    is_gpu = True
print("is gpu "+str(is_gpu))
"""

encoder_path = os.environ["encoder_path"]
print(encoder_path)
sys.path.insert(1,encoder_path)
from encoder import AEEncoder

decoder_path = os.environ["decoder_path"]
sys.path.insert(1,decoder_path)
from decoder import AEDecoder

class Eval():
    def __init__(self, param_dict):
        self.train_data = param_dict["train_data"]
        self.data_obj = DataProcessing(self.train_data,param_dict['prior_knowledge'],128,param_dict['relationships_filter'])
        self.encoder = AEEncoder(data=self.data_obj,dropout_rate=0,batch_norm=0,width_multiplier=param_dict['width_multiplier'],depth=param_dict['encoder_depth'])
        self.decoder = AEDecoder(data=self.data_obj,dropout_rate=0,batch_norm=0,width_multiplier=param_dict['width_multiplier'],depth=param_dict['decoder_depth'])
        self.model = AE(self.encoder,self.decoder)
        self.model.load_state_dict(torch.load(param_dict['model_path']))
        self.model.eval()

    def pred_TF_activities(self, input_data_path):
        input_data, input_tensor = self.load_input_data(input_data_path)
        output = self.model.encoder(input_tensor)
        return output
        
    def get_reconstruction(self, input_data_path):
        input_data, input_tensor = self.load_input_data(input_data_path)
        output = self.model(input_tensor)
        return output

    def load_input_data(self, input_data_path):
        input_data = pd.read_pickle(self.input_data_path)
        input_data = input_data.loc[:,self.data_obj.input_genes]
        input_tensor = torch.tensor(input_data.to_numpy()).float()
        return input_data, input_tensor

    def run_pert_tests(self, ko_data_path):
        auc, activity_df, ranked_df, ko_tf_ranks = get_ko_roc_curve(self.data_obj,ko_data_path,self.model.encoder,os.getcwd()+'/outputs/')

    def run_ko_tests(self, ko_data_path):
        auc, activity_df, ranked_df, ko_tf_ranks = get_knocktf_ko_roc_curve(self.data_obj,ko_data_path,self.model.encoder,os.getcwd()+'/outputs/')

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
    evaluation = Eval(params)
    #evaluation.pred_TF_activities(evaluation.input_data)
    evaluation.run_pert_tests("/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/pos_neg_samples/")
    evaluation.run_ko_tests("/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/pos_neg_samples/")
