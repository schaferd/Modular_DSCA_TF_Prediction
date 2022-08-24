import torch
import sys
import os

train_path = '/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/'
sys.path.insert(1,train_path)
from data_processing import DataProcessing

decoder_path = train_path+'gene_grouped_indep/'
sys.path.insert(1,decoder_path)
from gene_grouped_indep import GeneGroupedIndep as SparsityClass


def eval_sparsity(input_data,sparse_data,batch_size,relationships_filter):
    data_obj = DataProcessing(input_data,sparse_data,batch_size,relationships_filter)
    sparse_obj = SparsityClass(data_obj)
    
    #first_layer = sparse_obj.final_layer
    #middle_layer = sparse_obj.middle_layers
    #final_layer = sparse_obj.first_layer

    first_layer = sparse_obj.first_layer
    #middle_layer = sparse_obj.middle_layers
    final_layer = sparse_obj.final_layer

    first_layer = [first_layer[1],first_layer[0]]
    #middle_layer = [middle_layer[1],middle_layer[0]]
    final_layer = [final_layer[1],final_layer[0]]



    print((max(first_layer[0])+1,max(first_layer[1])+1))
    #print(max(middle_layer[0])+1,max(middle_layer[1])+1)
    print(max(final_layer[0])+1,max(final_layer[1])+1)

    first_matrix = torch.sparse_coo_tensor(first_layer,torch.ones(len(first_layer[0])),(max(first_layer[0])+1,max(first_layer[1])+1)).to_dense()
    #middle_matrix = torch.sparse_coo_tensor(middle_layer,torch.ones(len(middle_layer[0])),(max(middle_layer[0])+1,max(middle_layer[1])+1)).to_dense()
    final_matrix = torch.sparse_coo_tensor(final_layer,torch.ones(len(final_layer[0])),(max(final_layer[0])+1,max(final_layer[1])+1)).to_dense()


    tf_dict = {tf:v for v,tf in enumerate(data_obj.tfs)}
    gene_dict = {gene:v for v,gene in enumerate(data_obj.genes)}
    probe = torch.eye(len(data_obj.tfs))
    exp = torch.zeros(len(data_obj.tfs),len(data_obj.genes))
    for tf in enumerate(data_obj.tfs):
        tf = tf[1]
        genes = data_obj.tf_gene_dict[tf].keys()
        gene_indices = [gene_dict[gene] for gene in genes]
        tf_index = tf_dict[tf]
        for j in gene_indices:
            exp[tf_index][j] = 1
        

    #result = ((probe@first_matrix)@middle_matrix)@final_matrix
    result = (probe@first_matrix)@final_matrix
    
    exp_nz = exp.nonzero()
    result_nz = result.nonzero()

    if torch.equal(exp_nz,result_nz):
        print("equal")
    else:
        print("not equal")


if __name__=='__main__':
    SPARSE_DATA='/nobackup/users/schaferd/ae_project_data/dorothea_tf_gene_relationship_knowledge/dorotheaSelectionA.tsv'
    INPUT_DATA='/home/schaferd/ae_project/curr_gpu/transcriptomics_autoencoder/input_data_processing/hdf_agg_data.pkl'
    eval_sparsity(INPUT_DATA,SPARSE_DATA,2,128)




