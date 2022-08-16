import numpy as np


def get_shallow_layer(data_obj):
        #tf layer to relationship layer
        xcoords = []
        ycoords = []

        counter = 0
        #loop through tfs
        for i,tf in enumerate(data_obj.tfs):
            #loop through relationships with specified tf
            for j,gene in enumerate(data_obj.tf_gene_dict[tf][0]):
                if gene in data_obj.genes:
                    xcoords.append(gene_dict[gene])
                    ycoords.append(i)
                    counter += 1

        shallow_layer = np.vstack((np.array(xcoords),np.array(ycoords)))

        print("shallow layer complete")
        print(shallow_layer.shape)
        return shallow_layer
