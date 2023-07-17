import torch
from torch import nn

import numpy as np 
import pandas as pd
import collections


class AEEncoder(nn.Module):
        def __init__(self,**kwargs):
                super().__init__()

                self.data_obj = kwargs["data"]	
                self.width_multiplier = kwargs["width_multiplier"]
                self.depth = kwargs["depth"]

                self.tf_size = self.data_obj.tfs.size
                self.gene_size = len(self.data_obj.gene_names)
                self.dropout_rate = kwargs["dropout_rate"]
                #self.input_size = len(self.data_obj.input_data)

                mid_layer_size = self.gene_size*self.width_multiplier
                activ_func = nn.SELU()
                #activ_func = nn.LeakyReLU()

                encoder = collections.OrderedDict()

                encoder['do_1'] = nn.Dropout(self.dropout_rate)

                encoder['encoder_1'] = nn.Linear(self.gene_size,mid_layer_size)
                encoder['encoder_activ1'] = activ_func

                for i in range(2,self.depth+2):
                    encoder['do_'+str(i)] = nn.Dropout(0.1)
                    encoder['encoder_'+str(i)] = nn.Linear(mid_layer_size,mid_layer_size)
                    encoder['encoder_activ'+str(i)] = activ_func

                encoder['embedding'] = nn.Linear(mid_layer_size,self.tf_size)
                encoder['embedding_activ'] = activ_func

                self.ae_encoder = nn.Sequential(encoder)

        def forward(self,features):
                return self.ae_encoder(features)
