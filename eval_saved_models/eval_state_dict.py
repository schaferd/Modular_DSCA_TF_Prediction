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

is_gpu = False
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    is_gpu = True
print("is gpu "+str(is_gpu))

encoder_path = os.environ["encoder_path"]
print(encoder_path)
sys.path.insert(1,encoder_path)
from encoder import AEEncoder

decoder_path = os.environ["decoder_path"]
sys.path.insert(1,decoder_path)
from decoder import AEDecoder

class EvalStateDict():
    def __init__(self, param_dict):
        self.train_data = param_dict["train_data"]
        self.data_obj = DataProcessing(self.train_data,param_dict['prior_knowledge'],128,param_dict['relationships_filter'])
        self.encoder = AEEncoder(data=self.data_obj,dropout_rate=0,batch_norm=0,width_multiplier=param_dict['width_multiplier'],depth=param_dict['encoder_depth'])
        self.decoder = AEDecoder(data=self.data_obj,dropout_rate=0,batch_norm=0,width_multiplier=param_dict['width_multiplier'],depth=param_dict['decoder_depth'])
        self.model = AE(self.encoder,self.decoder).to(device)
        self.model.load_state_dict(param_dict['model'])
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

    def load_input_data(self, input_data_path):
        input_data = pd.read_pickle(input_data_path)
        try:
            input_data = input_data.loc[:,self.data_obj.input_genes]
        except:
            overlap_genes = list(set(self.data_obj.input_genes).intersection(set(input_data.columns)))
            overlap_genes.sort()
            input_data = input_data.loc[:,overlap_genes]
            input_data = pd.concat([pd.DataFrame(columns=self.data_obj.input_genes),input_data],axis=0).astype(float)
        print("input data")
        print(input_data)
        print(self.data_obj.input_genes)
        input_tensor = torch.tensor(input_data.to_numpy()).float().to(device)
        return input_data, input_tensor

    def run_pert_tests(self, ko_data_path,save_path):
        auc, activity_df, ranked_df, ko_tf_ranks, koed_tfs = get_ko_roc_curve(self.data_obj,ko_data_path,self.model.encoder,save_path)
        return auc

    def run_ko_tests(self, ko_data_path, save_path):
        auc, activity_df, ranked_df, ko_tf_ranks, koed_tfs = get_knocktf_ko_roc_curve(self.data_obj,ko_data_path,self.model.encoder,save_path)
        print(activity_df)
        print(koed_tfs)
        return auc, activity_df, koed_tfs

