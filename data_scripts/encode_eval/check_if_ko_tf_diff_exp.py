import pandas as pd
import numpy as np
from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)


ko_tfs = set()
with open('ensembl_tfs_ko.txt','r') as f:
    for line in f:
        line = line.strip()
        ko_tfs.add(line)

control_df = pd.read_pickle('control_df.pkl')
treat_df = pd.read_pickle('treat_df.pkl')
input_genes = '/nobackup/users/schaferd/ae_project_data/hdf_gene_expression_data/gene_input_data.pkl'
input_genes = pd.read_pickle(input_genes)
sample_id_to_tf = pd.read_pickle('sample_id_to_tf.pkl')
tf_to_sample_id = pd.read_pickle('tf_to_sample_id.pkl')

#check if ko tf in input genes
input_set = set(input_genes)
for tf in ko_tfs:
    if tf not in input_set:
        print("not in input genes")

gene_to_ensembl = {}
ensembl_to_gene = {}
for tf in tf_to_sample_id.keys():
        print(tf)
        try:
                ensembl_id = ensembl_data.gene_ids_of_gene_name(tf)
                print(ensembl_id)
                gene_to_ensembl[tf] = ensembl_id[0]
                ensembl_to_gene[ensembl_id[0]] = tf
        except:
                gene_to_ensembl[tf] = None

print(ensembl_to_gene)

#check if differentially expressed
for tf in ko_tfs:
    sample = tf_to_sample_id[ensembl_to_gene[tf]]
    c = control_df.loc[sample,tf]
    t = treat_df.loc[sample,tf]
    print(sample,tf)
    print(c,t)
    print(c-t)






