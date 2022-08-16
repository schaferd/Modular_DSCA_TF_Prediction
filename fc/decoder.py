import torch
from torch import nn

import numpy as np 
import pandas as pd
from gaussian_noise import GaussianNoise
import collections

class AEDecoder(nn.Module):
        def __init__(self,**kwargs):
                super().__init__()

                self.data_obj = kwargs["data"]	
                self.width_multiplier = kwargs["width_multiplier"]

                self.tf_size = self.data_obj.tfs.size
                self.gene_size = len(self.data_obj.gene_names)
                self.input_size = len(self.data_obj.input_data)
                self.output_size = len(self.data_obj.overlap_list)

                mid_layer_size = self.gene_size*self.width_multiplier 
                activ_func = nn.ReLU()

                decoder = collections.OrderedDict()

                decoder['decoder_1'] = nn.Linear(self.tf_size,mid_layer_size)
                decoder['decoder_activ1'] = activ_func

                decoder['decoder_'+str(i+2)]=nn.Linear(mid_layer_size,mid_layer_size)
                decoder['decoder_activ'+str(i+2)] = activ_func

                decoder['output'] = nn.Linear(mid_layer_size,self.output_size)
				
                self.ae_decoder = nn.Sequential(decoder)	

        def forward(self,features):
            return self.ae_decoder(features)


