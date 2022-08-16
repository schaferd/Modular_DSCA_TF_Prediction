import numpy as np
import pandas as pd
import sys


class TFGroupedFCIndep():
		#tfs and genes should be in order that they will appear in input/weight matrices!
        def __init__(self,data_obj,nodes_per_tf=1):
                np.set_printoptions(threshold=1000)
                self.tfs = data_obj.tfs
                self.genes = data_obj.genes 
                self.tf_gene_dict = data_obj.tf_gene_dict
                self.gene_tf_dict = data_obj.gene_tf_dict

                self.nodes_per_tf = nodes_per_tf
                
                self.tf_dict = {tf: v for v,tf in enumerate(self.tfs)}
                self.coord_tf_dict = {} #map each coord to its end gene
                counter = 0

                #loop through each tf
                for gene in self.genes:
                    #loop through each relationship that tf is in
                    for tf in self.gene_tf_dict[gene][0]:
                        if tf in self.tf_dict:
                            #repeat that relationship nodes_per_tf times
                            for i in range(nodes_per_tf):
                                #maps relationship coord to gene coord
                                self.coord_tf_dict[counter] = self.tf_dict[tf]
                                counter += 1 

                self.tf_features = [i for i in range(0,counter)]

                self.tf_coord_sets = {self.tf_dict[tf]: [] for tf in self.tfs} #map each gene to a list of all coords that will connect to that gene
                for f in self.coord_tf_dict:
                    tf = self.coord_tf_dict[f]
                    self.tf_coord_sets[tf].append(f)

                self.first_layer = self.get_first_layer()
                self.middle_layers = self.get_middle_layer()
                self.final_layer = self.get_final_layer()


        def get_first_layer(self):
                #tf layer to relationship layer
                xcoords = []
                ycoords = []

                counter = 0
                #loop through tfs
                for i,gene in enumerate(self.genes):
                    #loop through relationships with specified tf
                    for j,tf in enumerate(self.gene_tf_dict[gene][0]):
                        if tf in self.tfs:
                            #repeat that relationship in network nodes_per_tf times
                            for i_ in range(self.nodes_per_tf):
                                xcoords.append(counter)
                                ycoords.append(i)
                                counter += 1

                first_tf_layer = np.vstack((np.array(xcoords),np.array(ycoords)))
                return first_tf_layer
						
        def get_middle_layers(self):
                #relationship layer fully connected to relationship layer with same genes
                #return sparse weight matrix

                coords = None
                xcoords = []
                ycoords = []
                
                #loop through each gene
                for tf in self.tf_coord_sets:
                        #list with all node coords that gene is associated with
                        curr_list = self.tf_coord_sets[tf]
                        #make fully connected layer between relationships with same gene
                        for i in curr_list:
                            for j in curr_list:
                                xcoords.append(j)
                                ycoords.append(i)
									
                middle_tf_layer = np.vstack((np.array(xcoords),np.array(ycoords)))
                return middle_tf_layer 

        def get_final_layer(self):
                #relationship layer to gene layer
                coords = None
                xcoords = []
                ycoords = []

                #loop through each gene
                for tf in self.tf_coord_sets:
                    for i in self.tf_coord_sets[tf]:
                        xcoords.append(tf)
                        ycoords.append(i)

                final_tf_layer = np.vstack((np.array(xcoords),np.array(ycoords)))
                return final_tf_layer 


if __name__ == "__main__":
    for_loop_deep = ForLoopDeep()

