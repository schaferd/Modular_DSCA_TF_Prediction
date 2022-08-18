import torch
from torch.utils.checkpoint import checkpoint_sequential
from torch import nn

import numpy as np 
import pandas as pd
import sparselinear as sl
import collections

from tf_grouped_fc_indep import TFGroupedFCIndep

class AEDecoder(nn.Module):
        def __init__(self,**kwargs):
                super().__init__()

                self.data_obj = kwargs["data"]	

                self.dropout_rate = kwargs["dropout_rate"]
                self.is_bn = kwargs["batch_norm"]
                self.width_multiplier = kwargs["width_multiplier"]

                matrices_obj = TFGroupedFCIndep(self.data_obj,nodes_per_tf=self.width_multiplier)

                self.first_weights = matrices_obj.get_first_layer()
                self.middle_weights = matrices_obj.get_middle_layers()
                self.final_weights = matrices_obj.get_final_layer()

                self.first_weights = [self.final_weights[1],self.final_weights[0]]
                self.final_weights = [self.first_weights[1],self.first_weights[0]]
                    

                self.decoder_features = max(np.amax(self.first_weights,axis=0)[0]+1,np.amax(self.middle_weights.T,axis=0)[0]+1)
                self.tf_size = len(self.data_obj.tfs)
                self.gene_size = len(self.data_obj.input_genes)

                activ_func = nn.LeakyReLU()

                decoder = collections.OrderedDict()

                decoder['decoder_1'] = sl.SparseLinear(self.tf_size,self.decoder_features,connectivity=torch.tensor(self.first_weights))
                decoder['decoder_activ1'] = activ_func
                #if self.is_bn:
                #    decoder['bn_decoder1'] = nn.BatchNorm1d(self.decoder_features)


                decoder['decoder_2'] = sl.SparseLinear(self.decoder_features,self.decoder_features,connectivity=torch.tensor(self.middle_weights))
                decoder['decoder_activ2'] = activ_func
                #if self.is_bn:
                #    decoder['bn_decoder'+str(i+2)] = nn.BatchNorm1d(self.decoder_features)
                #if self.dropout_rate > 0:
                #    middle_layers['do_decoder'+str(i+2)] = nn.Dropout(self.dropout_rate)


                decoder['output'] = sl.SparseLinear(self.decoder_features,self.gene_size,connectivity=torch.tensor(self.final_weights))
                #decoder['output_activ'] = activ_func
                #if self.is_bn:
                #    decoder['bn_output'] = nn.BatchNorm1d(self.gene_size)
                #if self.dropout_rate > 0:
                #    final_layer['do_output'] = nn.Dropout(self.dropout_rate)
				
                self.decoder= nn.Sequential(decoder)
				


        def forward(self,features):
            return self.decoder(features)
