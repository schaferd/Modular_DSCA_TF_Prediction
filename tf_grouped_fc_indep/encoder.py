import torch
from torch.utils.checkpoint import checkpoint_sequential
from torch import nn

import numpy as np 
import pandas as pd
import sparselinear as sl
import collections

from tf_grouped_fc_indep import TFGroupedFCIndep

class AEEncoder(nn.Module):

        def __init__(self,**kwargs):
                super().__init__()


                self.data_obj = kwargs["data"]	

                self.dropout_rate = kwargs["dropout_rate"]
                self.is_bn = kwargs["batch_norm"]
                self.width_multiplier = kwargs["width_multiplier"]

                matrices_obj = TFGroupedFCIndep(self.data_obj,nodes_per_tf=self.width_multiplier)

                self.first_weights = matrices_obj.first_layer
                self.middle_weights = matrices_obj.middle_layers
                self.final_weights = matrices_obj.final_layers
                    
                activ_func = nn.LeakyReLU()

                self.encoder_features = np.amax(self.middle_weights.T,axis=0)[0]+1
                self.tf_size = np.amax(self.first_weights.T,axis=0)[1]+1
                self.gene_size = np.amax(self.final_weights.T,axis=0)[0]+1
                self.labels_size = len(self.data_obj.labels.columns)

                encoder = collections.OrderedDict()

                if self.dropout_rate > 0:
                    encoder['do_encoder1'] = nn.Dropout(self.dropout_rate)
                encoder['encoder_1'] = sl.SparseLinear(max(self.first_tf_weights[1])+1,max(self.first_tf_weights[0])+1,connectivity=torch.tensor(self.first_tf_weights))
                if self.is_bn:
                    encoder['bn_encoder1'] = nn.BatchNorm1d(max(self.final_weights[1])+1,affine=False)
                encoder['encoder_activ1'] = activ_func

                encoder['encoder_'+str(i+2)] = sl.SparseLinear(max(self.middle_tf_nz[1])+1,max(self.middle_tf_nz[0])+1,connectivity=torch.tensor(self.middle_tf_nz))
                if self.is_bn:
                    encoder['bn_encoder'+str(i+2)] = nn.BatchNorm1d(self.encoder_features,affine=False)
                encoder['encoder_activ'+str(i+2)] = activ_func
                #if self.dropout_rate > 0:
                #    middle_layers['do_encoder'+str(i+2)] = nn.Dropout(self.dropout_rate)

                encoder['embedding'] = sl.SparseLinear(max(self.final_tf_weights[1])+1,max(self.final_tf_weights[0])+1,connectivity=torch.tensor(self.final_tf_weights))
                if self.is_bn:
                    encoder['bn_embedding'] = nn.BatchNorm1d(self.tf_size,affine=False)
                encoder['embedding_activ'] = activ_func
                if self.gn > 0:
                    encoder['gb_embedding'] = GaussianNoise(self.tf_size,self.gn)

                self.encoder = nn.Sequential(encoder)



        def forward(self,features):
            return self.encoder(features)
