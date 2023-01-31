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

                matrices_obj = TFGroupedIndep(self.data_obj,nodes_per_tf=self.width_multiplier)

                self.first_weights = matrices_obj.first_layer
                self.final_weights = matrices_obj.final_layer

                print("enc first layer",max(self.first_weights[1])+1,max(self.first_weights[0])+1)
                print("enc last layers",max(self.final_weights[1])+1,max(self.final_weights[0])+1)

                activ_func = nn.LeakyReLU()

                encoder = collections.OrderedDict()

                if self.dropout_rate > 0:
                    encoder['do_encoder1'] = nn.Dropout(self.dropout_rate)
                encoder['encoder_1'] = sl.SparseLinear(max(self.first_weights[1])+1,max(self.first_weights[0])+1,connectivity=torch.tensor(self.first_weights))
                if self.is_bn:
                    encoder['bn_encoder1'] = nn.BatchNorm1d(max(self.first_weights[0])+1,affine=False)
                encoder['encoder_activ1'] = activ_func

                encoder['embedding'] = sl.SparseLinear(max(self.final_weights[1])+1,max(self.final_weights[0])+1,connectivity=torch.tensor(self.final_weights))
                if self.is_bn:
                    encoder['bn_embedding'] = nn.BatchNorm1d(max(self.final_weights[0])+1,affine=False)
                encoder['embedding_activ'] = activ_func
                #encoder['embedding_activ'] = nn.Sigmoid()

                self.encoder = nn.Sequential(encoder)



        def forward(self,features):
            return self.encoder(features)
