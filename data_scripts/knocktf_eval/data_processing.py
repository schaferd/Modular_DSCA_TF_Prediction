import pandas as pd
from scipy import stats
import numpy as np
from pyensembl import EnsemblRelease
import matplotlib.pyplot as plt
ensembl_data = EnsemblRelease(78)

ko_df = pd.read_csv("knocktf.txt",delimiter='\t',low_memory=False)
#ko_df = pd.read_csv("short_knocktf.txt",delimiter='\t',low_memory=False)
overlap = pd.read_csv('overlapping_exp.txt', delimiter=', ',engine='python') 
overlap_datasets = set(overlap["knocktf_ko"])
input_genes = '/nobackup/users/schaferd/ae_project_data/hdf_gene_expression_data/gene_input_data.pkl'

input_genes = pd.read_pickle(input_genes)
sample_ids = set(list(ko_df['Sample_ID']))

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
ko_df.drop(ko_df.loc[ko_df["Sample_ID"].isin(overlap_datasets)].index, inplace=True)
print(ko_df)

print(ko_df[ko_df["Sample_ID"]=='DataSet_01_119'])

#df['Mean Expr. of Control'] = pd.to_numeric(df['Mean Expr. of Control'])
#df['Mean Expr. of Treat'] = pd.to_numeric(df['Mean Expr. of Treat'])

#df['Control'] = (df['Mean Expr. of Control'] - df['Mean Expr. of Control'].mean())/df['Mean Expr. of Control'].std()
#df['Treat'] = (df['Mean Expr. of Treat'] - df['Mean Expr. of Treat'].mean())/df['Mean Expr. of Treat'].std()
"""
id_tf_df = ko_df.loc[:,["Sample_ID","TF"]]
id_tf_df = id_tf_df.groupby(['Sample_ID','TF']).size().reset_index().rename(columns={0:'count'})
id_tf_df.to_csv("id_tf.csv",sep='\t')
"""

new_ko_df = ko_df.drop(columns=["TF","FC","Log2FC","Rank","P_value","up_down"])
control_df = new_ko_df.copy(deep=True)
treat_df = new_ko_df.copy(deep=True)
control_df['Control'] = pd.to_numeric(control_df['Mean Expr. of Control'], errors='coerce')
control_df = control_df.drop(columns=['Mean Expr. of Control','Mean Expr. of Treat'])
treat_df['Treat'] = pd.to_numeric(treat_df['Mean Expr. of Treat'], errors='coerce')
treat_df = treat_df.drop(columns=['Mean Expr. of Control','Mean Expr. of Treat'])



#control_df = control_df.pivot(index="Sample_ID", columns="Gene")
#treat_df = treat_df.pivot(index="Sample_ID", columns="Gene")

def log_trans(df):
    #log transform
    df = np.log10(df+1)
    return df

def zscore(df):
    df = df.apply(lambda x: (x-x.mean())/x.std(), axis=0)
    return df

def cpm(df):
    for i in df.index:
        s = df.loc[i,:].sum()
        print(s)
        df.loc[i,:] = df.loc[i,:].apply(lambda x: ((10**6)*x)/s)
    return df

microarray_control = control_df[control_df['Sample_ID'].str.contains('DataSet_01')].pivot(index="Sample_ID", columns="Gene")
rna_seq_control = control_df[control_df['Sample_ID'].str.contains('DataSet_02')].pivot(index="Sample_ID", columns="Gene")
microarray_treated = treat_df[treat_df['Sample_ID'].str.contains('DataSet_01')].pivot(index="Sample_ID", columns="Gene")
rna_seq_treated = treat_df[treat_df['Sample_ID'].str.contains('DataSet_02')].pivot(index="Sample_ID", columns="Gene")

microarray_control.columns = microarray_control.columns.droplevel()
rna_seq_control.columns = rna_seq_control.columns.droplevel()
microarray_treated.columns = microarray_treated.columns.droplevel()
rna_seq_treated.columns = rna_seq_treated.columns.droplevel()

#rna_seq = pd.concat([rna_seq_control,rna_seq_treated],axis=0).fillna(0)
#microarray = pd.concat([microarray_control,microarray_treated],axis=0).fillna(0)
rna_seq = pd.concat([rna_seq_control,rna_seq_treated],axis=0)
microarray = pd.concat([microarray_control,microarray_treated],axis=0)

rna_seq = zscore(rna_seq)
microarray = zscore(microarray)

overlap_genes = list(set(input_genes).intersection(microarray.columns))
overlap_genes.sort()
temp_df = pd.DataFrame(columns=input_genes)
microarray = microarray.loc[:,overlap_genes]
#microarray = pd.concat([temp_df,microarray],axis=0).fillna(0)
microarray = pd.concat([temp_df,microarray],axis=0)

overlap_genes = list(set(input_genes).intersection(set(rna_seq.columns)))
overlap_genes.sort()
rna_seq = rna_seq.loc[:,overlap_genes]
#rna_seq = pd.concat([temp_df,rna_seq],axis=0).fillna(0)
rna_seq = pd.concat([temp_df,rna_seq],axis=0)

print("microarray")
print(microarray)
print("rna seq")
print(rna_seq)

microarray_control = microarray.iloc[:len(microarray_control.index),:]
microarray_treated = microarray.iloc[len(microarray_control.index):,:]
rna_seq_control = rna_seq.iloc[:len(rna_seq_control.index),:]
rna_seq_treated = rna_seq.iloc[len(rna_seq_control.index):,:]

print("rna seq control")
print(rna_seq_control)
print("rnq seq treated")
print(rna_seq_treated)

#control_df = pd.concat([microarray_control,rna_seq_control],axis=0).fillna(0)
#treat_df = pd.concat([microarray_treated,rna_seq_treated],axis=0).fillna(0)
control_df = pd.concat([microarray_control,rna_seq_control],axis=0)
treat_df = pd.concat([microarray_treated,rna_seq_treated],axis=0)

print("control")
print(control_df)

print("treated")
print(treat_df)

treat_df.to_pickle("treated_nan.pkl")
control_df.to_pickle("control_nan.pkl")
raise ValueError()

#treat_df.to_csv("treated.csv", sep='\t')
#control_df.to_csv("control.csv",sep='\t')

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

print('treat')
print(treat_df)
print('control')
print(control_df)

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
"""
save_files = {}
for col in df.columns:
    f = col[:-1]
    if f not in save_files:
        save_files[f] = [df[col].T]
    else:
        save_files[f].append(df[col].T)
print(save_files)

for f,dfs in save_files.items():
    print(dfs)
    df_dict = {i:series for i,series in enumerate(dfs)}
    df = pd.DataFrame(df_dict,index = dfs[0].index).T
    print("hello")
    print(df)
    df.to_pickle(f)
"""
    



#treated_df.to_csv(sample+"."+str(TF)+".treated.csv")
#control_df.to_csv(sample+"."+str(TF)+".control.csv")
