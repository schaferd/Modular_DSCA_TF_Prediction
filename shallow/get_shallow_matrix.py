import numpy as np


class Shallow():
    def __init__(self,data_obj):
        self.gene_dict = {g: v for v, g in enumerate(data_obj.genes)}
        self.data_obj = data_obj
        self.tfs = self.data_obj.tfs
        self.genes = self.data_obj.genes 
        self.tf_gene_dict = self.data_obj.tf_gene_dict

        self.gene_dict = {g: v for v, g in enumerate(self.genes)}
        self.tf_dict = {tf: i for i,tf in enumerate(self.tfs)}
        self.shallow_layer = self.get_shallow_layer()

    def get_shallow_layer(self):
            #tf layer to relationship layer
            xcoords = []
            ycoords = []
            coords = set() 

            for tf,genes in self.tf_gene_dict.items():
                for g in genes:
                    xcoords.append(self.gene_dict[g])
                    ycoords.append(self.tf_dict[tf])
                    coords.add((self.gene_dict[g],self.tf_dict[tf]))


            """
            #loop through tfs
            for i,tf in enumerate(self.tfs):
                #loop through relationships with specified tf
                for j,gene in enumerate(self.tf_gene_dict[tf][0]):
                    if gene in self.genes:
                        xcoords.append(self.gene_dict[gene])
                        ycoords.append(i)
                        coords.add((self.gene_dict[gene],i))
            """
                        

            shallow_layer = [xcoords,ycoords]

            return shallow_layer
