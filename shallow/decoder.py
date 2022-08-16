import torch
from torch import nn

import sparselinear as sl
import collections

from get_shallow_matrix import get_shallow_layer

class AEDecoder(nn.Module):

        def __init__(self,**kwargs):
                super().__init__()

                if len(kwargs) > 1:

                        self.data_obj = kwargs["data"]	

                        self.dropout_rate = kwargs["dropout_rate"]
                        self.is_bn = kwargs["batch_norm"]

                        self.shallow_matrix = get_shallow_layer(data_obj)
                        self.shallow_matrix = [self.shallow_matrix[1],self.shallow_matrix[0]]

                        activ_func = nn.LeakyReLU()
                        
                        decoder = collections.OrderedDict()
                        decoder['decoder_1'] = sl.SparseLinear(max(self.shallow_matrix[1])+1,max(self.shallow_matrix[0])+1,connectivity=torch.tensor(self.shallow_matrix))
                        #if self.is_bn:
                        #    decoder['bn_decoder1'] = nn.BatchNorm1d(max(self.shallow_matrix[1])+1,affine=False)
                        decoder['decoder_activ1'] = activ_func

                        self.decoder = nn.Sequential(decoder)
                else:
                        decoder = kwargs['state_dict']
                        self.ae_decoder = nn.Sequential(decoder)



        def forward(self,features):
                return self.decoder(features)
