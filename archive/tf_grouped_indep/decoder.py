import torch
from torch.utils.checkpoint import checkpoint_sequential
from torch import nn

import numpy as np 
import pandas as pd
import sparselinear as sl
import collections

from tf_grouped_indep import TFGroupedIndep

class AEDecoder(nn.Module):
        def __init__(self,**kwargs):
                super().__init__()

                self.data_obj = kwargs["data"]	

                self.dropout_rate = kwargs["dropout_rate"]
                self.is_bn = kwargs["batch_norm"]
                self.width_multiplier = kwargs["width_multiplier"]

                matrices_obj = TFGroupedIndep(self.data_obj,nodes_per_tf=self.width_multiplier)

                self.first_weights = [matrices_obj.final_layer[1],matrices_obj.final_layer[0]]
                self.final_weights = [matrices_obj.first_layer[1],matrices_obj.first_layer[0]]
                #self.first_weights = matrices_obj.final_layer
                #self.final_weights = matrices_obj.first_layer

                activ_func = nn.LeakyReLU()

                decoder = collections.OrderedDict()
                print("first layer",max(self.first_weights[1])+1,max(self.first_weights[0])+1)
                print("last layers",max(self.final_weights[1])+1,max(self.final_weights[0])+1)

                decoder['decoder_1'] = sl.SparseLinear(max(self.first_weights[1])+1,max(self.first_weights[0])+1,connectivity=torch.tensor(self.first_weights))
                decoder['decoder_activ1'] = activ_func
                #if self.is_bn:
                #    decoder['bn_decoder1'] = nn.BatchNorm1d(self.decoder_features)

                decoder['output'] = sl.SparseLinear(max(self.final_weights[1])+1,max(self.final_weights[0])+1,connectivity=torch.tensor(self.final_weights))
                #decoder['output_activ'] = activ_func
                #if self.is_bn:
                #    decoder['bn_output'] = nn.BatchNorm1d(self.gene_size)
                #if self.dropout_rate > 0:
                #    final_layer['do_output'] = nn.Dropout(self.dropout_rate)
				
                self.decoder= nn.Sequential(decoder)
				


        def forward(self,features):
            return self.decoder(features)
