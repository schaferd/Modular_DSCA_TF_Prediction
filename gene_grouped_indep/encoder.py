import torch
from torch.utils.checkpoint import checkpoint_sequential
from torch import nn

import numpy as np 
import pandas as pd
import sparselinear as sl
import collections

from gene_grouped_indep import GeneGroupedIndep


class AEEncoder(nn.Module):

        def __init__(self,**kwargs):
                super().__init__()

                self.data_obj = kwargs["data"]	

                self.decoder_depth = kwargs["decoder_depth"]
                self.encoder_depth = kwargs["encoder_depth"]
                self.dropout_rate = kwargs["dropout_rate"]
                self.is_bn = kwargs["batch_norm"]
                self.gn = kwargs["gaussian_noise"]
                self.width_multiplier = kwargs["width_multiplier"]

                matrices_obj = GeneGroupedIndep(self.data_obj,nodes_per_gene=self.width_multiplier)

                self.first_weights = matrices_obj.first_layer 
                self.final_weights = matrices_obj.final_layer 

                self.first_weights = [self.final_weights[1],self.final_weights[0]]
                self.final_weights = [self.first_weights[1],self.first_weights[0]]
                    
                activ_func = nn.LeakyReLU()

                self.encoder_features = np.amax(self.middle_weights.T,axis=0)[0]+1
                self.tf_size = np.amax(self.first_weights.T,axis=0)[1]+1
                self.gene_size = np.amax(self.final_weights.T,axis=0)[0]+1
                self.labels_size = len(self.data_obj.labels.columns)

                encoder = collections.OrderedDict()


                if self.dropout_rate > 0:
                    encoder['do_encoder1'] = nn.Dropout(self.dropout_rate)
                first_layer['encoder_1'] = sl.SparseLinear(max(self.first_weights[1])+1,max(self.first_weights[0])+1,connectivity=torch.tensor(self.first_weights))
                if self.is_bn:
                    encoder['bn_encoder1'] = nn.BatchNorm1d(max(self.first_weights[1])+1,affine=False)
                first_layer['encoder_activ1'] = activ_func

                encoder['embedding'] = sl.SparseLinear(max(self.final_weights[1])+1,max(self.final_weights[0])+1,connectivity=torch.tensor(self.final_weights))
                if self.is_bn:
                    encoder['bn_embedding'] = nn.BatchNorm1d(self.tf_size,affine=False)
                encoder['embedding_activ'] = activ_func

                self.encoder = nn.Sequential(encoder)


        def forward(self,features):
                return self.encoder(features)
