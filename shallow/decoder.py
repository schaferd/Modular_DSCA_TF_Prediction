import torch
from torch import nn

import sparselinear as sl
import collections

from get_shallow_matrix import Shallow
from gene_grouped_indep import GeneGroupedIndep

class AEDecoder(nn.Module):

        def __init__(self,**kwargs):
                super().__init__()

                self.data_obj = kwargs["data"]	

                self.dropout_rate = kwargs["dropout_rate"]
                self.is_bn = kwargs["batch_norm"]

                shallow = Shallow(self.data_obj)
                self.shallow_matrix = shallow.shallow_layer

                activ_func = nn.LeakyReLU()
                
                decoder = collections.OrderedDict()
                print(torch.ops.torch_scatter.cuda_version())
                decoder['decoder_1'] = sl.SparseLinear(max(self.shallow_matrix[1])+1,max(self.shallow_matrix[0])+1,connectivity=torch.tensor(self.shallow_matrix).cpu())
                #if self.is_bn:
                #    decoder['bn_decoder1'] = nn.BatchNorm1d(max(self.shallow_matrix[1])+1,affine=False)
                #decoder['decoder_activ1'] = activ_func

                self.decoder = nn.Sequential(decoder)


        def forward(self,features):
                return self.decoder(features)
