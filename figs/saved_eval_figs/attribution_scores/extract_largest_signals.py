import pandas as pd
import matplotlib as mpl
import matplotlib
import random
import matplotlib.pyplot as plt
from time import sleep
import seaborn as sns
import os
import sys
import numpy as np
from load_attribution_scores import get_attr_scores_one_run, ensembl_to_gene_name, get_attr_scores_avg_runs
import requests 
import networkx as nx
from matplotlib.colors import ListedColormap, LinearSegmentedColormap



pknAB = pd.read_csv('pknAB.csv',sep='\t')
pknCD = pd.read_csv('pknCD.csv',sep='\t')
attr_df = get_attr_scores_one_run()
attr_df = ensembl_to_gene_name(attr_df).T

attr_df[attr_df.abs() < 0.004] = 0
indices = np.nonzero(attr_df)

tfs = attr_df.index[indices[0]]
genes = attr_df.columns[indices[1]]

pairs = np.array([tfs,genes]).T
counter = 0

in_PKN_tfs = []
in_PKN_genes = []

not_in_PKN_tfs = []
not_in_PKN_genes = []

for p in pairs:
    source = pknAB[pknAB['source'] == p[0]]
    sourceCD = pknCD[pknCD['source'] == p[0]]
    if p[1] in set(source['target']):
        #print(p)
        counter += 1
        in_PKN_tfs.append(p[0])
        in_PKN_genes.append(p[1])
    elif p[1] in set(sourceCD['target']):
        print("CD",p)
        counter += 1
        not_in_PKN_tfs.append(p[0])
        not_in_PKN_genes.append(p[1])
    else:
        print(p)
        not_in_PKN_tfs.append(p[0])
        not_in_PKN_genes.append(p[1])
#print(counter/len(pairs))
#print(len(pairs))

##FROM MATPLOTLIB.ORG
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-35, ha="left",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar
####END FROM MATPLOTLIB.ORG

##STRING DB

def get_string_network_image(filename,genes):
    string_api_url = "https://version-11-5.string-db.org/api"
    output_format = "image"
    method = "network"
    request_url = '/'.join([string_api_url,output_format, method])

    params = {
            "identifiers": "\r".join(genes),
            "species": 9606,
            "network_flavor":"evidence"
    }
    response = requests.post(request_url,data=params)
    with open(filename, 'wb') as fh:
        fh.write(response.content)

def get_string_network_df(genes):
    string_api_url = "https://version-11-5.string-db.org/api"
    output_format = "tsv"
    method = "network"
    request_url = '/'.join([string_api_url,output_format, method])
    params = {
        "identifiers" : "\r".join(genes), # your protein
        "species" : 9606, # species NCBI identifier
            "network_flavor":"evidence"
    }
    response = requests.post(request_url,data=params)
    print(response)
    cols = []
    df_dicts = []
    for i,line in enumerate(response.text.strip().split("\n")):
        l = line.strip().split('\t')
        if i == 0:
            cols = l
        else:
            row_dict = {}
            for j,c in enumerate(cols):
                row_dict[c] = l[j]
            df_dicts.append(row_dict)

    network_df = pd.DataFrame(df_dicts).drop_duplicates()
    return network_df

def is_path(geneA,geneB,network_df):
    queue = [geneA]
    explored = {geneA:0}
    while len(queue) > 0:
        curr_gene = queue.pop(0)
        if curr_gene == geneB:
            return True, explored[geneB]
        neighbors = get_neighbors(curr_gene,network_df)
        for g in neighbors:
            if g not in explored:
                explored[g] = explored[curr_gene]+1
                queue.append(g)
    return False, 0

def get_cc(gene,network_df):
    queue = [gene]
    explored = set([gene])
    while len(queue) > 0:
        print(explored)
        curr_gene = queue.pop(0)
        neighbors = get_neighbors(curr_gene,network_df)
        for g in neighbors:
            if g not in explored:
                explored.add(g)
                queue.append(g)
    return explored

def is_direct_connection(geneA,geneB,network_df):
    neighbors = set(get_neighbors(geneA,network_df))
    if geneB in neighbors:
        return True
    return False



def get_neighbors(gene,network_df):
    geneA_indices = network_df[network_df['preferredName_A'] == gene]['preferredName_B']
    geneB_indices = network_df[network_df['preferredName_B'] == gene]['preferredName_A']
    neighbors = list(set(np.hstack([geneA_indices,geneB_indices])))
    return neighbors

def probe_random_relationships(num_pairs):
    genes = pknAB.sample(n=num_pairs,replace=True)['target']
    tfs = pknAB.sample(n=num_pairs, replace=True)['source']
    pairs = np.vstack([genes,tfs]).T
    network = get_string_network_df(list(np.hstack([list(genes),list(tfs)])))
    sleep(0.9)
    #return do_pairs_exist(pairs,network)
    return num_direct_connections(pairs,network)

def do_pairs_exist(pairs,network):
    exists_evidence = []
    path_lengths = []
    for pair in pairs: 
        has_path, path_length = is_path(pair[0],pair[1],network)
        exists_evidence.append(has_path)
        if has_path:
            path_lengths.append(path_length)
    return np.array(path_lengths).mean()*(1-(sum(exists_evidence)/len(exists_evidence)))

def num_direct_connections(pairs, network):
    connections = []
    for pair in pairs: 
        connections.append(is_direct_connection(pair[0],pair[1],network))
    return sum(connections)/len(connections)


#pd.to_pickle(not_in_PKN_tfs,'not_in_PKN_tfs.pkl')
#pd.to_pickle(not_in_PKN_genes,'not_in_PKN_genes.pkl')
not_in_PKN_tfs = pd.read_pickle('not_in_PKN_tfs.pkl')
not_in_PKN_genes = pd.read_pickle('not_in_PKN_genes.pkl')

attr_pairs = np.vstack([not_in_PKN_tfs,not_in_PKN_genes]).T
attr_pairs_df = pd.DataFrame({'tf':not_in_PKN_tfs,'gene':not_in_PKN_genes})
attr_genes = list(np.hstack([not_in_PKN_tfs,not_in_PKN_genes]))
network = get_string_network_df(attr_genes)
attr_dir_connections = num_direct_connections(attr_pairs,network)

AB_pairs_df = pd.DataFrame({'tf':in_PKN_tfs,'gene':in_PKN_genes})



plt.clf()
random_dist = np.array([probe_random_relationships(16) for i in range(2000)])
pd.to_pickle(random_dist,"random_rels_dist.pkl")

print("pval",len(random_dist[random_dist >= attr_dir_connections])/len(random_dist))

print(random_dist)
plt.hist(random_dist)
plt.savefig('random_rel_hist.png')
plt.clf()
raise ValueError()

def is_pred_relationship(geneA,geneB,rels):
    if geneA in set(rels['tf']):
        if geneB in set(rels[rels['tf'] == geneA]['gene']):
            return True
    elif geneA in set(rels['gene']):
        if geneB in set(rels[rels['gene'] == geneA]['tf']):
            return True
    return False

def is_in_CDpkn(geneA,geneB):
    if geneA in set(pknCD['source']):
        sourceCD = pknCD[pknCD['source'] == geneA]
        if geneB in set(sourceCD['target']):
            return True
    elif geneA in set(pknCD['target']):
        sourceCD = pknCD[pknCD['target'] == geneA]
        if geneB in set(sourceCD['source']):
            return True
    return False

def is_in_ABpkn(geneA,geneB):
    if geneA in set(pknAB['source']):
        sourceAB = pknAB[pknAB['source'] == geneA]
        if geneB in set(sourceAB['target']):
            return True
    elif geneA in set(pknAB['target']):
        sourceAB = pknCD[pknAB['target'] == geneA]
        if geneB in set(sourceAB['source']):
            return True
    return False

def split_into_connected_components(df):
    not_discovered = set(np.hstack([df['preferredName_A'],df['preferredName_B']]))
    ccs = []
    while len(not_discovered) > 0:
        gene = random.sample(list(not_discovered),1)[0]
        print(gene)
        cc = get_cc(gene,df)
        ccs.append(list(cc))
        not_discovered = not_discovered - cc
    return ccs

def get_df_cc(cc,df):
    indices = []
    for gene in cc:
        if gene in set(df['preferredName_A']):
            indices.extend(df[df['preferredName_A'] == gene].index)

    return df.loc[indices,:].sort_values(by=['preferredName_A'])


def create_networkx_graph(df,rels,counter):
    graph = nx.Graph() 

    for i,row in df.iterrows():
        color = 'black'
        weight = 1
        geneA = row['preferredName_A']
        geneB = row['preferredName_B']
        if is_in_CDpkn(geneA,geneB):
            color = 'orange'
            weight = 4
        elif is_pred_relationship(geneA,geneB,rels):
            color = 'pink'
            weight = 4

        graph.add_edge(geneA,geneB,color=color,weight=weight)

    seed = 0
    #pos = nx.spring_layout(graph.subgraph(cc),seed=seed,scale=4)
    pos = nx.circular_layout(graph.subgraph(cc),scale=4)

    edges = graph.edges()
    colors = [graph[u][v]['color'] for u,v in edges]
    weights = [graph[u][v]['weight'] for u,v in edges]

    plt.clf()
    #pos = nx.spring_layout(graph,scale=1.5,seed=6,k=0.5)

    nx.draw(graph,pos=pos,edge_color=colors,width=weights,with_labels=True,node_color='skyblue',node_size=1000,font_size=10)
    plt.savefig('test_graph_cc'+str(counter)+'.png')

"""
print(network)
ccs = split_into_connected_components(network)
print(ccs)
counter = 0
for cc in ccs:
    print(counter)
    df = get_df_cc(cc,network)
    create_networkx_graph(df,attr_pairs_df,counter)
    counter += 1
"""
def is_in_stringdb(tf,gene,network):
    if tf in set(network['preferredName_A']):
        if gene in set(network[network['preferredName_A'] == tf]['preferredName_B']):
            return True
    elif tf in set(network['preferredName_B']):
        if gene in set(network[network['preferredName_B'] == tf]['preferredName_A']):
            return True
    return False


def create_adjacency_df(pairs,network,ax):
    rows = list(np.unique(pairs['tf']))
    columns = list(np.unique(pairs['gene']))
    overlap_AB_pairs = AB_pairs_df[(AB_pairs_df['tf'].isin(rows)) & (AB_pairs_df['gene'].isin(columns))]
    pairs = pd.concat([pairs,overlap_AB_pairs])

    rows.sort()
    columns.sort()
    labels = ['AB','CD','StringDB','No Evidence', 'Not Predicted']
    norm = matplotlib.colors.BoundaryNorm([0.5,1.5,2.5,3.5,4.5,5.5],5)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[::-1][norm(x)])
    type_dict = {'AB':4.5,'CD':3.5,'StringDB':2.5,'No Evidence':1.5,'Not Predicted':0.5}
    df = pd.DataFrame(np.zeros((len(rows),len(columns))),index=rows,columns=columns)
    for i,row in pairs.iterrows():
        tf = row['tf']
        gene = row['gene']
        if is_in_ABpkn(tf,gene):
            df.loc[tf,gene] = type_dict['AB']
        elif is_in_CDpkn(tf,gene):
            df.loc[tf,gene] = type_dict['CD']
        elif is_in_stringdb(tf,gene,network):
            df.loc[tf,gene] = type_dict['StringDB']
        else:
            df.loc[tf,gene] = type_dict['No Evidence']
    print(df)
    #clustergrid = sns.clustermap(df)
    #plt.clf()
    #df = df.iloc[clustergrid.dendrogram_row.reordered_ind,clustergrid.dendrogram_col.reordered_ind]
    #colors = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
    colors = ["mediumpurple", "plum", "skyblue", "0.7", "0.9"]
    colors.reverse()
    my_cmap = ListedColormap(colors, name="my_cmap")
    heatmap(df,df.index,df.columns,ax=ax, norm=norm,cmap=my_cmap, cbar_kw=dict(ticks=np.arange(0,6),format=fmt),cbarlabel="Evidence Type")
    ax.set_ylabel('TF')
    ax.set_xlabel('Gene')
    return df

def get_heatmap(ax):
    df = create_adjacency_df(attr_pairs_df,network,ax)

def create_bipartite_rel_graph(pairs,network):
    graph = nx.Graph() 
    print(pairs)
    pairs.sort_values(by=['tf'])
    for i, row in pairs.iterrows():
        color = '0.73' 
        weight = 1
        tf = row['tf']
        gene = row['gene']
        if is_in_CDpkn(tf,gene):
            color = 'orange'
            weight = 4
        elif is_in_stringdb(tf,gene,network):
            color = 'turquoise'
            weight = 4
        graph.add_edge(tf,gene,color=color,weight=weight)

    pos = nx.bipartite_layout(graph,pairs['tf'],align='vertical',aspect_ratio=5)
    edges = graph.edges()
    colors = [graph[u][v]['color'] for u,v in edges]
    weights = [graph[u][v]['weight'] for u,v in edges]

    plt.clf()
    nx.draw(graph,pos=pos,edge_color=colors,width=weights,with_labels=True,node_color='lavender',node_size=700,font_size=10)
    plt.savefig('test_graph_bipartite.png')

create_bipartite_rel_graph(attr_pairs_df,network)
