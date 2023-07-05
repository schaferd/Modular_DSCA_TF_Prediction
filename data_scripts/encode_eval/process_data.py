import pandas as pd
import numpy as np
import os 
import sys
from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)
import matplotlib.pyplot as plt

def process_file(file_path):
    df = pd.read_csv(file_path,sep='\t',index_col=0)
    keep_rows = [i for i in df.index if 'ENSG' in i]
    df = df.loc[keep_rows,:]
    new_df = df.loc[:,['TPM']].T
    return new_df

def log_trans(df):
    #log transform
    df = np.log10(df+1)
    return df

def zscore(df):
    df = df.apply(lambda x: (x-x.mean())/x.std(), axis=0)
    return df

data_dir = "/nobackup/users/schaferd/ae_project_data/encode_ko_data/raw_counts/"
input_genes = '/nobackup/users/schaferd/ae_project_data/hdf_gene_expression_data/gene_input_data.pkl'
input_genes = pd.read_pickle(input_genes)
control_paths = {}
treated_paths = {}

for i in os.listdir(data_dir):
    control_paths[i] = [data_dir+i+'/control/'+j for j in os.listdir(data_dir+i+'/control/')]
    treated_paths[i] = [data_dir+i+'/treated/'+j for j in os.listdir(data_dir+i+'/treated/')]

control_df = pd.DataFrame(columns = ['Sample ID','TF','gene_id','TPM'])
treat_df = pd.DataFrame(columns = ['Sample ID','TF','gene_id','TPM'])

sample_id_to_tf = {}
tf_to_sample_id = {}

control_dfs = []

"""
for exp in control_paths:
    df_pair = []
    for path in control_paths[exp]:
        df = process_file(path)
        sample_id = exp.split('.')[0]
        tf = exp.split('.')[1]
        genes = df.columns
        sample_id_to_tf[sample_id] = tf
        tf_to_sample_id[tf] = sample_id
        ensembl_ids = {i.split('.')[0]:[] for i in genes}
        [ensembl_ids[i.split('.')[0]].append(df.loc['TPM',i]) for i in genes]
        #ensembl_ids = {i:max(ensembl_ids[i]) for i in ensembl_ids}
        ensembl_ids = {i:sum(ensembl_ids[i]) for i in ensembl_ids}
        new_df = pd.DataFrame(ensembl_ids,index=['TPM']).T.reset_index().rename(columns={'index':'gene_id'})
        new_df['Sample ID'] = sample_id
        new_df['TF'] = tf

        df_pair.append(new_df)

    np_pair = np.array([df['TPM'].to_numpy() for df in df_pair])
    avg_df = df_pair[0]
    avg_df['TPM'] = np_pair.mean(axis=0)
    control_dfs.append(avg_df)
    print(len(control_dfs))

control_df = pd.concat(control_dfs).reset_index(drop=True)
control_df = control_df.pivot(index='Sample ID',columns='gene_id',values='TPM')
print(control_df)
#control_df.to_pickle('control_df.pkl')
pd.to_pickle(sample_id_to_tf,'sample_id_to_tf.pkl')
pd.to_pickle(tf_to_sample_id,'tf_to_sample_id.pkl')

treat_dfs = []

for exp in treated_paths:
    df_pair = []
    for path in treated_paths[exp]:
        df = process_file(path)
        sample_id = exp.split('.')[0]
        tf = exp.split('.')[1]
        genes = df.columns
        ensembl_ids = {i.split('.')[0]:[] for i in genes}
        [ensembl_ids[i.split('.')[0]].append(df.loc['TPM',i]) for i in genes]
        #ensembl_ids = {i:max(ensembl_ids[i]) for i in ensembl_ids}
        ensembl_ids = {i:sum(ensembl_ids[i]) for i in ensembl_ids}
        new_df = pd.DataFrame(ensembl_ids,index=['TPM']).T.reset_index().rename(columns={'index':'gene_id'})
        new_df['Sample ID'] = sample_id
        new_df['TF'] = tf

        df_pair.append(new_df)
    np_pair = np.array([df['TPM'].to_numpy() for df in df_pair])
    avg_df = df_pair[0]
    avg_df['TPM'] = np_pair.mean(axis=0)
    treat_dfs.append(avg_df)
    print(len(treat_dfs))
    
treat_df = pd.concat(treat_dfs).reset_index(drop=True)
treat_df = treat_df.pivot(index='Sample ID',columns='gene_id',values='TPM')
print(treat_df)
#treat_df.to_pickle('treat_df.pkl')
"""

def min_nonzero(row):
    row_vals = np.array(row)
    non_zeros = row_vals[np.nonzero(row_vals)]
    return min(non_zeros)


def clr(df):
    for i,row in df.iterrows():
        min_val = min_nonzero(row)/2
        row = row.apply(lambda x: np.log2(x+min_val))
        #row = row.apply(lambda x: np.log2(x+1))
        mean = row.mean()
        row = row.apply(lambda x: x - mean)
        df.loc[i,:] = row
    return df

def zscore(df,mean,std):
    df = df.subtract(mean,axis=1)
    df = df.divide(std,axis=1)
    return df

sample_id_to_tf = pd.read_pickle('sample_id_to_tf.pkl')
tf_to_sample_id = pd.read_pickle('tf_to_sample_id.pkl')

control_df = pd.read_pickle('control_df.pkl').fillna(0)
treat_df = pd.read_pickle('treat_df.pkl').fillna(0)

print(control_df)

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

per_ranks_ko_tfs = []

"""
for sample in sample_id_to_tf.keys():
    gene = sample_id_to_tf[sample]
    row = gene_id_treat_df.loc[sample,:].argsort()
    standing = row.loc[gene]/len(row)
    per_ranks_ko_tfs.append(standing)
    if standing > 0.4:
        control_df = control_df.drop([sample])
        treat_df = treat_df.drop([sample])
"""

print(control_df)

control_df  = clr(control_df)
treat_df = clr(treat_df)

mean_of_control_per_gene = control_df.mean(axis=0)
std_of_control_per_gene = control_df.std(axis=0)

mean_of_treat_per_gene = treat_df.mean(axis=0)
std_of_treat_per_gene = treat_df.std(axis=0)

treat_df = zscore(treat_df,mean_of_control_per_gene,std_of_control_per_gene).fillna(0)
#treat_df = zscore(treat_df,mean_of_treat_per_gene,std_of_treat_per_gene).fillna(0)
control_df = zscore(control_df,mean_of_control_per_gene,std_of_control_per_gene).fillna(0)

#treat_df.to_pickle('processed_treat_df.pkl')
#control_df.to_pickle('processed_control_df.pkl')

#treat_df = pd.read_pickle('processed_treat_df.pkl')
#control_df = pd.read_pickle('processed_control_df.pkl')

overlap_genes = list(set(input_genes).intersection(control_df.columns))
overlap_genes.sort()
temp_df = pd.DataFrame(columns=input_genes)
control_df = control_df.loc[:,overlap_genes]
control_df = pd.concat([temp_df,control_df],axis=0).fillna(0)
treat_df = treat_df.loc[:,overlap_genes]
treat_df = pd.concat([temp_df,treat_df],axis=0).fillna(0)


id_tf = pd.read_pickle('sample_id_to_tf.pkl')
tfs = []
ids = []
for s_id in id_tf:
    ids.append(s_id)
    tfs.append(id_tf[s_id])
id_tf_df = pd.DataFrame({'Sample_ID':ids,'TF':tfs})
id_tf_df.to_csv('id_tf.csv',sep='\t')

#control_df = pd.read_pickle('filtered_processed_control_df.pkl')
#treat_df = pd.read_pickle('filtered_processed_treat_df.pkl')

treat_df.to_csv('filtered_processed_treat_df.csv',sep='\t')
control_df.to_csv('filtered_processed_control_df.csv',sep='\t')

combined_df = pd.concat([treat_df,control_df],axis=0)
combined_df.to_pickle('combined_processed_df.pkl')
raise ValueError()



control_df = control_df.rename(columns=ensembl_to_gene)
treat_df = treat_df.rename(columns=ensembl_to_gene)


for sample_id in list(treat_df.index):
    sample = treat_df[treat_df.index == sample_id]
    sample.index.name = 'Sample_ID'
    tf = sample_id_to_tf[sample_id] 
    file_name = 'ko_datafiles/'+sample_id+'.'+tf+'.'+'treated.csv'
    print(file_name)
    sample.to_csv(file_name)
    

for sample_id in control_df.index:
    sample = treat_df[treat_df.index == sample_id]
    sample.index.name = 'Sample_ID'
    tf = sample_id_to_tf[sample_id] 
    file_name = 'ko_datafiles/'+sample_id+'.'+tf+'.'+'control.csv'
    print(file_name)
    sample.to_csv(file_name)
    



