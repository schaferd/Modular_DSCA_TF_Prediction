import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample

def get_overlap_matrix(data):
    print(data)
    samples = data.index
    result_matrix = pd.DataFrame(0,columns=samples,index=samples)
    pairwise_comb = list(itertools.combinations(samples,2))
    for s1,s2 in pairwise_comb:
        overlap = get_overlap(s1,s2,data)
        print(s1,s2,overlap)
        result_matrix.iloc[s1,s2] = overlap
        result_matrix.iloc[s2,s1] = overlap
    sns.clustermap(result_matrix,cbar_kws={'label':'# of genes in overlap'})
    plt.title('Input Gene Overlap Across Samples')
    plt.xlabel('Samples')
    plt.ylabel('Samples')
    plt.savefig('overlap_fig.png')

def get_overlap(s1,s2,data):
    df = data.iloc[[s1,s2]].dropna(axis=1)
    return len(df.columns)


if __name__ == '__main__':
    data_path1 = pd.read_pickle('roc_agg_data_nan.pkl')#.sample(n=500)
    #data_path2 = pd.read_pickle('hdf_agg_data.pkl').iloc[:10]
    #full_df = pd.concat([data_path1,data_path2],ignore_index=True)
    #get_overlap_matrix(full_df)
    get_overlap_matrix(data_path1)
