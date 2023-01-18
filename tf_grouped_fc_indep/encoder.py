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
                self.depth = kwargs["depth"]

                matrices_obj = TFGroupedFCIndep(self.data_obj,nodes_per_tf=self.width_multiplier)

                self.first_weights = matrices_obj.first_layer
                self.middle_weights = matrices_obj.middle_layers
                self.final_weights = matrices_obj.final_layer
                    
                #activ_func = nn.LeakyReLU()
                activ_func = nn.SELU()

                encoder = collections.OrderedDict()

                if self.dropout_rate > 0:
                    encoder['do_encoder1'] = nn.Dropout(self.dropout_rate)
                encoder['encoder_1'] = sl.SparseLinear(max(self.first_weights[1])+1,max(self.first_weights[0])+1,connectivity=torch.tensor(self.first_weights))
                if self.is_bn:
                    encoder['bn_encoder1'] = nn.BatchNorm1d(max(self.first_weights[0])+1,affine=False)
                encoder['encoder_activ1'] = activ_func

                for i in range(2,self.depth+2):
                    encoder['encoder_'+str(i)] = sl.SparseLinear(max(self.middle_weights[1])+1,max(self.middle_weights[0])+1,connectivity=torch.tensor(self.middle_weights))
                    encoder['encoder_activ'+str(i)] = activ_func
                    if self.is_bn:
                        encoder['bn_encoder'+str(i)] = nn.BatchNorm1d(max(self.middle_weights[0])+1,affine=False)
                    #if self.dropout_rate > 0:
                    #    middle_layers['do_encoder'+str(i)] = nn.Dropout(self.dropout_rate)

                encoder['embedding'] = sl.SparseLinear(max(self.final_weights[1])+1,max(self.final_weights[0])+1,connectivity=torch.tensor(self.final_weights))
                encoder['embedding_activ'] = activ_func
                if self.is_bn:
                    encoder['bn_embedding'] = nn.BatchNorm1d(max(self.final_weights[0])+1,affine=False)

                self.encoder = nn.Sequential(encoder)



        def forward(self,features):
            return self.encoder(features)
