import torch
from torch.utils.checkpoint import checkpoint_sequential
from torch import nn

import numpy as np 
import pandas as pd
import sparselinear as sl
import collections

from tf_grouped_indep import TFGroupedIndep

class AEEncoder(nn.Module):

        def __init__(self,**kwargs):
                super().__init__()


                self.data_obj = kwargs["data"]	

                self.dropout_rate = kwargs["dropout_rate"]
                self.is_bn = kwargs["batch_norm"]
                self.width_multiplier = kwargs["width_multiplier"]

                matrices_obj = TFGroupedIndep(data_obj,nodes_per_tf=self.width_multiplier)

                self.first_weights = matrices_obj.get_first_layer()
                self.final_weights = matrices_obj.get_final_layers()
                    
                activ_func = nn.LeakyReLU()

                self.encoder_features = np.amax(self.middle_weights.T,axis=0)[0]+1
                self.tf_size = np.amax(self.first_weights.T,axis=0)[1]+1
                self.gene_size = np.amax(self.final_weights.T,axis=0)[0]+1
                self.labels_size = len(self.data_obj.labels.columns)

                encoder = collections.OrderedDict()

                if self.dropout_rate > 0:
                    first_layer['do_encoder1'] = nn.Dropout(self.dropout_rate)
                first_layer['encoder_1'] = sl.SparseLinear(max(self.first_tf_weights[1])+1,max(self.first_tf_weights[0])+1,connectivity=torch.tensor(self.first_tf_weights))
                if self.is_bn:
                    first_layer['bn_encoder1'] = nn.BatchNorm1d(max(self.final_weights[1])+1,affine=False)
                first_layer['encoder_activ1'] = activ_func

                final_layer['embedding'] = sl.SparseLinear(max(self.final_tf_weights[1])+1,max(self.final_tf_weights[0])+1,connectivity=torch.tensor(self.final_tf_weights))
                if self.is_bn:
                    final_layer['bn_embedding'] = nn.BatchNorm1d(self.tf_size,affine=False)
                final_layer['embedding_activ'] = activ_func
                #encoder['embedding_activ'] = nn.Sigmoid()

                self.encoder = nn.Sequential(encoder)



        def forward(self,features):
            return self.encoder(features)
