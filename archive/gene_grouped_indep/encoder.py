import torch
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

                self.dropout_rate = kwargs["dropout_rate"]
                self.is_bn = kwargs["batch_norm"]
                self.width_multiplier = kwargs["width_multiplier"]

                matrices_obj = GeneGroupedIndep(self.data_obj,nodes_per_gene=self.width_multiplier)

                self.first_weights = [matrices_obj.final_layer[1],matrices_obj.final_layer[0]] 
                self.final_weights = [matrices_obj.first_layer[1],matrices_obj.first_layer[0]]

                #activ_func = nn.LeakyReLU()
                activ_func=nn.Tanh()

                encoder = collections.OrderedDict()


                if self.dropout_rate > 0:
                    encoder['do_encoder1'] = nn.Dropout(self.dropout_rate)
                encoder['encoder_1'] = sl.SparseLinear(max(self.first_weights[1])+1,max(self.first_weights[0])+1,connectivity=torch.tensor(self.first_weights))
                encoder['encoder_activ1'] = nn.ReLU()
                if self.is_bn:
                    encoder['bn_encoder1'] = nn.BatchNorm1d(max(self.first_weights[0])+1,affine=True)

                encoder['embedding'] = sl.SparseLinear(max(self.final_weights[1])+1,max(self.final_weights[0])+1,connectivity=torch.tensor(self.final_weights))
                #encoder['embedding_activ'] = nn.ReLU()
                if self.is_bn:
                    encoder['bn_embedding'] = nn.BatchNorm1d(max(self.final_weights[0])+1,affine=True)

                #encoder['embedding_activ'] = nn.ReLU()
                self.encoder = nn.Sequential(encoder)


        def forward(self,features):
                return self.encoder(features)
