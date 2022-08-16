import numpy as np
import pandas as pd
import sys


class GeneGroupedFCIndep():
	#tfs and genes should be in order that they will appear in input/weight matrices!
        def __init__(self,data_obj,nodes_per_gene=1):
                self.data_obj = data_obj
                self.tfs = self.data_obj.tfs
                self.genes = self.data_obj.genes 
                self.tf_gene_dict = self.data_obj.tf_gene_dict
                self.gene_tf_dict = self.data_obj.gene_tf_dict

                self.coord_gene_dict = {} #map each coord to its end gene
                self.nodes_per_gene = nodes_per_gene
                
                #creates a coord for each gene
                self.gene_dict = {g: v for v, g in enumerate(genes)}
                counter = 0
                
                #loop through each tf
                for tf in self.tfs:
                    #loop through each relationship that tf is in
                    for gene in sparse_dict[tf][0]:
                        if gene in self.gene_dict:
                            #repeat that relationship nodes_per_gene times
                            for i in range(nodes_per_gene):
                                #maps relationship coord to gene coord
                                self.coord_gene_dict[counter] = self.gene_dict[gene]
                                counter += 1 

                self.features = [i for i in range(0,counter)]

                self.gene_coord_sets = {self.gene_dict[g]: [] for g in genes} #map each gene to a list of all coords that will connect to that gene
                for f in self.coord_gene_dict:
                    g = self.coord_gene_dict[f]
                    self.gene_coord_sets[g].append(f)

                self.first_layer = self.get_first_layer()
                self.middle_layer = self.get_middle_layers() 
                self.final_layer = self.get_final_layer() 


        def get_first_layer(self):
                #tf layer to relationship layer
                moa = []

                xcoords = []
                ycoords = []

                counter = 0
                #loop through tfs
                for i,tf in enumerate(self.tfs):
                    #loop through relationships with specified tf
                    for j,gene in enumerate(self.sparse_dict[tf][0]):
                        if gene in self.genes:
                            #repeat that relationship in network nodes_per_gene times
                            for i_ in range(self.nodes_per_gene):
                                xcoords.append(counter)
                                ycoords.append(i)
                                moa.append(self.sparse_dict[tf][1][j])
                                counter += 1

                first_layer = np.vstack((np.array(xcoords),np.array(ycoords)))

                return first_layer
						

        def get_middle_layers(self):
                #relationship layer fully connected to relationship layer with same genes
                #return sparse weight matrix

                coords = None
                xcoords = []
                ycoords = []
                
                #loop through each gene
                for g in self.gene_coord_sets:
                        #list with all node coords that gene is associated with
                        curr_list = self.gene_coord_sets[g]
                        #make fully connected layer between relationships with same gene
                        for i in curr_list:
                            for j in curr_list:
                                xcoords.append(j)
                                ycoords.append(i)
									
                middle_layer = np.vstack((np.array(xcoords),np.array(ycoords)))
                return middle_layer 


        def get_final_layer(self):
                #relationship layer to gene layer
                coords = None
                xcoords = []
                ycoords = []

                #loop through each gene
                for g in self.gene_coord_sets:
                    for i in self.gene_coord_sets[g]:
                        xcoords.append(g)
                        ycoords.append(i)

                final_layer = np.vstack((np.array(xcoords),np.array(ycoords)))
                return final_layer 



if __name__ == "__main__":
    for_loop_deep = ForLoopDeep()

