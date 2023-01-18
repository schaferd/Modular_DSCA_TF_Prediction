import torch
from torch.utils.checkpoint import checkpoint_sequential
from torch import nn

import numpy as np 
import pandas as pd
import sparselinear as sl
import collections

from gene_grouped_fc_indep import GeneGroupedFCIndep


class AEDecoder(nn.Module):
        def __init__(self,**kwargs):
                super().__init__()

                self.data_obj = kwargs["data"]	

                self.dropout_rate = kwargs["dropout_rate"]
                self.is_bn = kwargs["batch_norm"]
                self.width_multiplier = kwargs["width_multiplier"]
                self.depth = kwargs["depth"]

                matrices_obj = GeneGroupedFCIndep(self.data_obj,nodes_per_gene=self.width_multiplier)

                self.first_weights = matrices_obj.first_layer 
                self.middle_weights = matrices_obj.middle_layer 
                self.final_weights = matrices_obj.final_layer 

                #activ_func = nn.Tanh()
                activ_func = nn.SELU()

                decoder = collections.OrderedDict()

                decoder['decoder_1'] = sl.SparseLinear(max(self.first_weights[1])+1,max(self.first_weights[0])+1,connectivity=torch.tensor(self.first_weights))
                decoder['decoder_activ1'] = activ_func
                #if self.is_bn:
                #    decoder['bn_decoder1'] = nn.BatchNorm1d(self.decoder_features)
				
                for i in range(2,self.depth+2):
                    decoder['decoder_'+str(i)] = sl.SparseLinear(max(self.middle_weights[1])+1,max(self.middle_weights[0])+1,connectivity=torch.tensor(self.middle_weights))
                    decoder['decoder_activ'+str(i)] = activ_func
                    #if self.is_bn:
                    #    decoder['bn_decoder'+str(i)] = nn.BatchNorm1d(self.decoder_features)
                    #if self.dropout_rate > 0:
                    #    middle_layers['do_decoder'+str(i)] = nn.Dropout(self.dropout_rate)

                #decoder['decoder_3'] = sl.SparseLinear(max(self.middle_weights[1])+1,max(self.middle_weights[0])+1,connectivity=torch.tensor(self.middle_weights))
                #decoder['decoder_activ3'] = nn.Tanh()

                decoder['output'] = sl.SparseLinear(max(self.final_weights[1])+1,max(self.final_weights[0])+1,connectivity=torch.tensor(self.final_weights))                
                #decoder['output_activ'] = activ_func
                #if self.is_bn:
                #    decoder['bn_output'] = nn.BatchNorm1d(self.labels_size)
                #if self.dropout_rate > 0:
                #    final_layer['do_output'] = nn.Dropout(self.dropout_rate)
				
                self.decoder = nn.Sequential(decoder)
				


        def forward(self,features):
            return self.decoder(features)

