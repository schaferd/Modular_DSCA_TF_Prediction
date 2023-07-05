import pandas as pd
import os
import numpy as np
from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)
import matplotlib.pyplot as plt

FILTER_CUTOFF = 0.4


id_tf = pd.read_csv("id_tf.csv",sep='\t',index_col=1)
id_tf = id_tf.drop(columns=['count','Unnamed: 0']).T
id_tf_dict = id_tf.to_dict(orient='list')
sample_to_tf = {i: id_tf_dict[i][0] for i in id_tf_dict}
tf_to_sample = {id_tf_dict[i][0]:i for i in id_tf_dict}

control_df = pd.read_csv("control.csv",sep='\t',index_col=0)
treat_df = pd.read_csv("treated.csv",sep='\t',index_col=0)
input_genes = '/nobackup/users/schaferd/ae_project_data/hdf_gene_expression_data/gene_input_data.pkl'
input_genes = pd.read_pickle(input_genes)

compatible_exps = set()
with open('compatible_exps.txt','r') as f:
    for line in f:
        compatible_exps.add(line.strip().split('.')[0])

gene_to_ensembl = {}
ensembl_to_gene = {}

for g in control_df.columns:
        try:
                gene_id = ensembl_data.gene_name_of_gene_id(g)
                gene_to_ensembl[gene_id] = g 
                ensembl_to_gene[g] = gene_id
        except:
                ensembl_to_gene[g] = None

gene_id_control_df = control_df.rename(columns=ensembl_to_gene)
gene_id_treat_df = treat_df.rename(columns=ensembl_to_gene)


def filter_by_diff_ranking(control_df,treat_df):
    per_ranks_ko_tfs = []
    for sample,row in gene_id_treat_df.iterrows():
        standing = 0
        if sample in compatible_exps:
            #CONTROL-TREATED RANKING
            ko_gene = sample_to_tf[sample]
            diff_row = gene_id_control_df.loc[sample]-row
            sort_row = diff_row.argsort()
            ko_rank = sort_row.loc[ko_gene]
            standing = ko_rank/len(row)

            per_ranks_ko_tfs.append(standing)
        if standing < 1-FILTER_CUTOFF:
            treat_df = treat_df.drop([sample])
            control_df = control_df.drop([sample])

    plt.hist(per_ranks_ko_tfs)
    plt.savefig('ko_tf_diff_rank_hist.png')
    print(control_df)
    print(treat_df)
    return control_df, treat_df

def filter_by_fc_ranking(control_df,treat_df):
    per_ranks_ko_tfs = []
    for sample,row in gene_id_treat_df.iterrows():
        standing = 0
        if sample in compatible_exps:
            #CONTROL-TREATED RANKING
            ko_gene = sample_to_tf[sample]
            div_row = gene_id_control_df.loc[sample]/row
            sort_row = div_row.argsort()
            ko_rank = sort_row.loc[ko_gene]
            standing = ko_rank/len(row)

            per_ranks_ko_tfs.append(standing)
        if standing < 1-FILTER_CUTOFF:
            treat_df = treat_df.drop([sample])
            control_df = control_df.drop([sample])

    plt.hist(per_ranks_ko_tfs)
    plt.savefig('ko_tf_fc_rank_hist.png')
    print(control_df)
    print(treat_df)
    return control_df, treat_df

def filter_by_treat_ranking(control_df,treated_df):
    per_ranks_ko_tfs = []
    for sample,row in gene_id_treat_df.iterrows():
        standing = 1
        if sample in compatible_exps:
            #TREATED RANKING
            sort_row = row.argsort()
            ko_gene = sample_to_tf[sample]
            ko_rank = sort_row.loc[ko_gene]
            standing = ko_rank/len(row)
            per_ranks_ko_tfs.append(standing)
        if standing > FILTER_CUTOFF:
            treat_df = treat_df.drop([sample])
            control_df = control_df.drop([sample])

    plt.hist(per_ranks_ko_tfs)
    plt.savefig('ko_tf_treat_rank_hist.png')
    print(control_df)
    print(treat_df)
    return control_df, treat_df

def filter_if_relevant(control_df,treat_df):
    for sample,row in gene_id_treat_df.iterrows():
        standing = 1
        if sample not in compatible_exps:
            treat_df = treat_df.drop([sample])
            control_df = control_df.drop([sample])
    return control_df, treat_df

#control_df, treat_df = filter_by_diff_ranking(control_df,treat_df)
control_df, treat_df = filter_by_fc_ranking(control_df,treat_df)
#control_df, treat_df = filter_if_relevant(control_df,treat_df)

#treat_df.to_csv("filtered_data/relevant_data/treated_relevant_samples.csv",sep='\t')
#control_df.to_csv("filtered_data/relevant_data/control_relevant_samples.csv",sep='\t')

treat_df.to_csv("filtered_data/fc_filtered/filtered_treated_"+str(FILTER_CUTOFF)+".csv", sep='\t')
control_df.to_csv("filtered_data/fc_filtered/filtered_control_"+str(FILTER_CUTOFF)+".csv",sep='\t')

control_df = control_df.rename(columns=ensembl_to_gene)
treat_df = treat_df.rename(columns=ensembl_to_gene)

print(control_df)
print(treat_df)

new_dir = 'filtered_data/fc_filtered/viper_data/filtered_'+str(FILTER_CUTOFF)+'/'
#new_dir = 'filtered_data/relevant_data/viper_data/samples/'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

    
for sample_id in list(treat_df.index):
    sample = treat_df[treat_df.index == sample_id]
    sample.index.name = 'Sample_ID'
    tf = sample_to_tf[sample_id]
    file_name = new_dir+sample_id+'.'+tf+'.'+'treated.csv'
    print(file_name)
    sample.to_csv(file_name)
    

for sample_id in control_df.index:
    sample = treat_df[treat_df.index == sample_id]
    sample.index.name = 'Sample_ID'
    tf = sample_to_tf[sample_id]
    file_name = new_dir+sample_id+'.'+tf+'.'+'control.csv'
    print(file_name)
    sample.to_csv(file_name)

print("done")

