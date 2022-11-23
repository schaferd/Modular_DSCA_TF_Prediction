import pandas as pd
import numpy as np


def get_overlapping_genes(ae_input_genes,genes):
    overlapping_genes = []
    genes = set(genes)
    for gene in ae_input_genes:
        if gene in genes:
            overlapping_genes.append(gene)
    return overlapping_genes

def convert_gene_name_to_ensembl(gene_name):
    try:
        ensembl_id = ensembl_data.gene_ids_of_gene_name(gene_name)
        return ensembl_id
    except:
        return [None]

if __name__ == '__main__':
    input_data = pd.read_csv('/nobackup/users/schaferd/ae_project_data/single_cell_blood/GSE94820_raw.expMatrix_DCnMono.discovery.set.submission.txt',header=0,sep='\t').T.transform(lambda x: np.log(x+1))
    input_data.to_pickle('/nobackup/users/schaferd/ae_project_data/single_cell_blood/agg_data.pkl')
    
    ensembl_genes = []
    for gene in input_data.columns:
        conv = convert_gene_name_to_ensembl(gene)
        if conv[0] is not None:
            input_data.rename(columns={gene:conv[0]})
        else:
            input_data.drop(gene,axis=1)

