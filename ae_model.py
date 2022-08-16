import torch
from torch import nn

class AE(nn.Module):
            def __init__(self,encoder,decoder,**kwargs):
                super().__init__()

                self.encoder = encoder 
                self.decoder = decoder 

                self.last_embedding = None

            def forward(self,features):
                embedding = self.encoder(features)
                reconstruction = self.decoder(embedding)
                self.last_embedding = embedding
                return reconstruction 
