import pandas as pd
from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)
import os
import pickle as pkl


def aggregate_data(data_dir):
    data_files = [data_dir+f for f in os.listdir(data_dir) if os.path.isfile(data_dir+f) and 'signature' in f]
    agg_df = None
    for f in data_files:
        print("data file",f)
        sample_data = load_sample_data(f)
        if agg_df is None:
            agg_df = sample_data 
        else:
            agg_df = pd.concat([agg_df,sample_data],ignore_index=True)
        print(len(agg_df)) 
        print(agg_df)
    #print("nan",agg_df.isna().sum().sum())
    #agg_df = agg_df.fillna(0)
    print(agg_df)
    agg_df.to_pickle(data_dir+'agg_data_nan.pkl')

def load_sample_data(f):
    file_list = []
    with open(f,'r') as f_:
        lines = f_.readlines()
        for line in lines:
            file_list.append(line)
    data = file_list
    gene_expression_dict = {}
    for i,row in enumerate(data):
        if i != 0:
            temp = row.split(' ')
            gene_expression_dict[temp[0].replace('"','')] = float(temp[-1].strip())
    df = pd.DataFrame(gene_expression_dict,index=[0])
    conv_dict = {}
    for gene in df.columns:
        genes = convert_gene_name_to_ensembl(gene)
        if genes[0] is not None:
            conv_dict[gene] = genes[0]
        else:
            df = df.drop(gene,axis=1)
    df = df.rename(columns=conv_dict)

    return df

def open_pkl(pkl_file):
    with open(pkl_file,'rb') as f:
            return pkl.load(f)

def convert_gene_name_to_ensembl(gene_name):
    try:
        ensembl_id = ensembl_data.gene_ids_of_gene_name(gene_name)
        return ensembl_id
    except:
        return [None]

if __name__ == "__main__":
    aggregate_data('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/contrasts/')
