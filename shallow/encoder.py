import torch
from torch import nn

import numpy as np 
import sparselinear as sl
import collections

from get_shallow_matrix import Shallow

class AEEncoder(nn.Module):

        def __init__(self,**kwargs):
                super().__init__()
                self.data_obj = kwargs["data"]	

                self.dropout_rate = kwargs["dropout_rate"]
                self.is_bn = kwargs["batch_norm"]

                shallow = Shallow(self.data_obj)
                self.shallow_matrix = shallow.shallow_layer
                self.shallow_matrix = [self.shallow_matrix[1],self.shallow_matrix[0]]

                activ_func = nn.LeakyReLU()
                
                encoder = collections.OrderedDict()
                if self.dropout_rate > 0:
                    encoder['do_encoder1'] = nn.Dropout(self.dropout_rate)
                encoder['encoder_1'] = sl.SparseLinear(max(self.shallow_matrix[1])+1,max(self.shallow_matrix[0])+1,connectivity=torch.tensor(self.shallow_matrix))
                if self.is_bn:
                    encoder['bn_encoder1'] = nn.BatchNorm1d(max(self.shallow_matrix[0])+1,affine=False)
                encoder['encoder_activ1'] = activ_func

                self.encoder = nn.Sequential(encoder)



        def forward(self,features):
                return self.encoder(features)
