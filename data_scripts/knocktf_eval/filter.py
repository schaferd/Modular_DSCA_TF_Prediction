import pandas as pd
import numpy as np
from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)

control_df = pd.read_csv('control.csv',sep='\t',index_col=0)
treat_df = pd.read_csv('treated.csv',sep='\t',index_col=0)

"""
print(control_df)
print(treat_df)

not_overlaps = []
for i,row in control_df.iterrows():
    if row.equals(treat_df.loc[i,:]) == False:
        not_overlaps.append(i)

print(len(not_overlaps))
control_df = control_df.loc[not_overlaps,:]
treat_df = treat_df.loc[not_overlaps,:]

print(control_df)
print(treat_df)

treat_df.to_csv("treated.csv", sep='\t')
control_df.to_csv("control.csv",sep='\t')
"""

microarray = []
rna_seq = []

for i, row in control_df.iterrows():
    if 'DataSet_01' in i:
        microarray.append(i)
    else:
        rna_seq.append(i)

control_df.loc[microarray,:].to_csv('microarray_control.csv',sep='\t')
control_df.loc[rna_seq,:].to_csv('rna_seq_control.csv',sep='\t')
treat_df.loc[microarray,:].to_csv('microarray_treated.csv',sep='\t')
treat_df.loc[rna_seq,:].to_csv('rna_seq_treated.csv',sep='\t')

raise ValueError()

ko_df = pd.read_csv("knocktf.txt",delimiter='\t',low_memory=False)
overlap = pd.read_csv('overlapping_exp.txt', delimiter=', ',engine='python') 
overlap_datasets = set(overlap["knocktf_ko"])

gene_to_ensembl = {}
ensembl_to_gene = {}

genes = ko_df["Gene"]
ensembl_genes = []
for g in genes:
        try:
                ensembl_id = ensembl_data.gene_ids_of_gene_name(g)
                ensembl_genes.append(ensembl_id[0])
                gene_to_ensembl[g] = ensembl_id[0]
                ensembl_to_gene[ensembl_id[0]] = g
        except:
                ensembl_genes.append(None) 
                gene_to_ensembl[g] = None

ko_df["Gene"] = ensembl_genes

ko_df =ko_df.dropna()
ko_df.drop(ko_df.loc[ko_df["Sample_ID"].isin(overlap)].index, inplace=True)

treat_overlap = list(set(ensembl_to_gene.keys()).intersection(set(treat_df.columns)))
control_overlap = list(set(ensembl_to_gene.keys()).intersection(set(control_df.columns)))

treat_overlap.sort()
control_overlap.sort()

treat_dict = {ens:ensembl_to_gene[ens] for ens in treat_overlap}
control_dict = {ens:ensembl_to_gene[ens] for ens in control_overlap}

treat_df = treat_df.loc[:,treat_overlap]
control_df = control_df.loc[:,control_overlap]
treat_df = treat_df.rename(treat_dict,axis='columns')
control_df = control_df.rename(control_dict,axis='columns')


for sample_id in list(treat_df.index):
    sample = treat_df[treat_df.index == sample_id]
    sample.index.name = 'Sample_ID'
    tf = list(ko_df[ko_df['Sample_ID'] == sample_id]['TF'])[0]
    file_name = 'ko_datafiles/'+sample_id+'.'+tf+'.'+'treated.csv'
    print(file_name)
    sample.to_csv(file_name)
    

for sample_id in control_df.index:
    sample = treat_df[treat_df.index == sample_id]
    sample.index.name = 'Sample_ID'
    tf = list(ko_df[ko_df['Sample_ID'] == sample_id]['TF'])[0]
    file_name = 'ko_datafiles/'+sample_id+'.'+tf+'.'+'control.csv'
    print(file_name)
    sample.to_csv(file_name)

print("done")
