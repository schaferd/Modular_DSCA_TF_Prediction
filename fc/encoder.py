import torch
from torch import nn

import numpy as np 
import pandas as pd
from gaussian_noise import GaussianNoise
import collections


class AEEncoder(nn.Module):
        def __init__(self,**kwargs):
                super().__init__()

                self.data_obj = kwargs["data"]	
                self.width_multiplier = kwargs["width_multiplier"]

                self.tf_size = self.data_obj.tfs.size
                self.gene_size = len(self.data_obj.gene_names)
                #self.input_size = len(self.data_obj.input_data)

                mid_layer_size = self.gene_size*self.width_multiplier
                activ_func = nn.ReLU()

                encoder = collections.OrderedDict()

                encoder['encoder_1'] = nn.Linear(self.gene_size,mid_layer_size)
                encoder['encoder_activ1'] = activ_func

                encoder['encoder_'+str(i+2)] = nn.Linear(mid_layer_size,mid_layer_size)
                encoder['encoder_activ'+str(i+2)] = activ_func

                encoder['embedding'] = nn.Linear(mid_layer_size,self.tf_size)
                encoder['embedding_activ'] = activ_func

                self.ae_encoder = nn.Sequential(encoder)

        def forward(self,features):
                return self.ae_encoder(features)
