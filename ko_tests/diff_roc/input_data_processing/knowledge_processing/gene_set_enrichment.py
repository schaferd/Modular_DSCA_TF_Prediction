import pandas as pd
import re
import csv
import numpy as np
from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)

def get_knowledge_dict(f):
    knowledge_dict = {}
    with open(f,'r') as csvfile:
        #datareader = csv.reader(csvfile,delimiter='\t')
        for row in csvfile:
            row = re.split(r'\t',row.strip())
            print(row)
            name = '_'.join(row[0].split('_')[:-2])
            gene_symbols = row[2:]
            print(gene_symbols)
            """
            ensembl_ids = []
            for gene in gene_symbols:
                if gene[:4] == 'ENSG':
                    ensembl_ids.append(gene)
                else:
                    ensembl = get_ensembl_id(gene)
                    ensembl_ids.extend(ensembl)

            if len(ensembl_ids) > 15:
                knowledge_dict[name]= ensembl_ids
            """
            knowledge_dict[name] = gene_symbols
    return knowledge_dict

def create_columns(knowledge_dict):
    tfs = []
    genes = []
    for tf,g in knowledge_dict.items():
        print(tf)
        for gene in g:
            tfs.append(tf)
            genes.append(gene)
    mor = [0 for i in tfs]
    return tfs, genes, mor

def save_data_df(tfs,genes,mor):
    data_dict = {'tf':tfs,'target':genes,'mor':mor}
    df = pd.DataFrame(data_dict)
    print(df)
    df.to_csv('gene_set.tsv',sep='\t')


def get_ensembl_id(gene_symbol):
    ensembl_ids = []
    try:
            ensembl_ids = ensembl_data.gene_ids_of_gene_name(gene_symbol)
    except:
            ensembl_ids = []
    return ensembl_ids

if __name__ == '__main__':
    f = '/nobackup/users/schaferd/ae_project_data/gene_set_enrichment_analysis/c3.tft.v7.5.1.symbols.gmt'
    knowledge_dict = get_knowledge_dict(f)
    tfs,genes,mor = create_columns(knowledge_dict)
    save_data_df(tfs,genes,mor)


