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
from load_attribution_scores import * #get_attr_scores_one_run, ensembl_to_gene_name, get_attr_scores_avg_runs
import requests 
import networkx as nx
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


pknAB = pd.read_csv('pknAB.csv',sep='\t')
pknCD = pd.read_csv('pknCD.csv',sep='\t')

rand_pknCD = pknCD.sample(frac=1)
rand_pknCD['source'] = pknCD.sample(frac=1)['source']

#attr_df = get_attr_scores_one_run()
#attr_df = ensembl_to_gene_name(attr_df).T
attr_dfs = [ensembl_to_gene_name(df).T for df in get_attr_scores_all_runs()]

for attr_df in attr_dfs:
    attr_df[attr_df.abs() < 0.004] = 0 
indices = [np.nonzero(attr_df) for attr_df in attr_dfs]

print(indices,attr_df)
tfs = [attr_df.index[indices[i][0]] for i,attr_df in enumerate(attr_dfs)]
genes = [attr_df.columns[indices[i][1]] for i,attr_df in enumerate(attr_dfs)]

pairs = [np.array([tfs[i],gs]).T for i, gs in enumerate(genes)]
#print(pairs)
#print(np.unique(np.array(pairs),axis=1,return_counts=True))
#raise ValueError()
counter = 0

all_models_in_PKN_tfs = []
all_models_in_PKN_genes = []
all_models_not_in_PKN_tfs = []
all_models_not_in_PKN_genes = []

for pair_set in pairs:
    in_PKN_tfs = []
    in_PKN_genes = []

    not_in_PKN_tfs = []
    not_in_PKN_genes = []

    for p in pair_set:
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
    all_models_in_PKN_tfs.append(in_PKN_tfs)
    all_models_in_PKN_genes.append(in_PKN_genes)
    all_models_not_in_PKN_tfs.append(not_in_PKN_tfs)
    all_models_not_in_PKN_genes.append(not_in_PKN_genes)
    

#print(counter/len(pairs))
#print(len(pairs))

##FROM MATPLOTLIB.ORG
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="",tick_labels=False, **kwargs):
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

    if tick_labels:
        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
        ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=False, bottom=True,
                       labeltop=False, labelbottom=True)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-35, ha="left",
                 rotation_mode="anchor")
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0)
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
    print(network_df)
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


pd.to_pickle(not_in_PKN_tfs,'not_in_PKN_tfs.pkl')
pd.to_pickle(not_in_PKN_genes,'not_in_PKN_genes.pkl')
not_in_PKN_tfs = pd.read_pickle('not_in_PKN_tfs.pkl')
not_in_PKN_genes = pd.read_pickle('not_in_PKN_genes.pkl')
pd.to_pickle(all_models_not_in_PKN_tfs, 'all_models_not_in_PKN_tfs.pkl')
pd.to_pickle(all_models_not_in_PKN_genes, 'all_models_not_in_PKN_genes.pkl')
pd.to_pickle(all_models_in_PKN_tfs, 'all_models_in_PKN_tfs.pkl')
pd.to_pickle(all_models_in_PKN_genes, 'all_models_in_PKN_genes.pkl')

all_models_not_in_PKN_tfs = pd.read_pickle('all_models_not_in_PKN_tfs.pkl')
all_models_not_in_PKN_genes = pd.read_pickle('all_models_not_in_PKN_genes.pkl')
all_models_in_PKN_tfs = pd.read_pickle('all_models_in_PKN_tfs.pkl')
all_models_in_PKN_genes = pd.read_pickle('all_models_in_PKN_genes.pkl')

attr_pairs = [np.vstack([not_in_PKN_tfs,all_models_not_in_PKN_genes[i]]).T for i,not_in_PKN_tfs in enumerate(all_models_not_in_PKN_tfs)]
attr_pairs_df = [pd.DataFrame({'tf':not_in_PKN_tfs,'gene':all_models_not_in_PKN_genes[i]}) for i,not_in_PKN_tfs in enumerate(all_models_not_in_PKN_tfs)]
attr_genes = [list(np.hstack([not_in_PKN_tfs,all_models_not_in_PKN_genes[i]])) for i,not_in_PKN_tfs in enumerate(all_models_not_in_PKN_tfs)]
network = [get_string_network_df(g) for g in attr_genes]
attr_dir_connections = [num_direct_connections(attr_pairs[i],n) for i,n in enumerate(network)]

AB_pairs_df = pd.concat([pd.DataFrame({'tf':in_PKN_tfs,'gene':all_models_in_PKN_genes[i]}) for i,in_PKN_tfs in enumerate(all_models_in_PKN_tfs)])

comb_pairs_df = [pd.DataFrame({'tf':all_models_not_in_PKN_tfs[i]+all_models_in_PKN_tfs[i],'gene':all_models_not_in_PKN_genes[i]+all_models_in_PKN_genes[i]}) for i in range(len(all_models_in_PKN_tfs))]
all_comb_pairs_df = pd.concat(comb_pairs_df)

all_networks = pd.concat(network)

all_attr_pairs_df = pd.concat(attr_pairs_df)
print(len(attr_pairs))
all_attr_pairs_unique = all_attr_pairs_df.groupby(all_attr_pairs_df.columns.tolist(),as_index=False).size()
all_attr_pairs_unique = all_attr_pairs_unique[all_attr_pairs_unique['size'] > 2]
comb_pairs_df_unique = all_comb_pairs_df.groupby(all_comb_pairs_df.columns.tolist(),as_index=False).size()
comb_pairs_df_filtered = comb_pairs_df_unique[comb_pairs_df_unique['size'] > 7]
all_networks = pd.concat(network)
print(all_attr_pairs_df)
print(all_networks)


"""
plt.clf()
random_dist = np.array([probe_random_relationships(16) for i in range(2000)])
pd.to_pickle(random_dist,"random_rels_dist.pkl")

print("pval",len(random_dist[random_dist >= attr_dir_connections])/len(random_dist))

print(random_dist)
plt.hist(random_dist)
plt.savefig('random_rel_hist.png')
plt.clf()
"""

def get_model_num_filter_line_plot(ax):
    pairs_df = comb_pairs_df_unique
    models_counts = {'AB':[],'CD':[],'StringDB':[],'No Evidence':[]}
    for m in range(1,max(list(pairs_df['size']))+1):
        rels = pairs_df[pairs_df['size'] >= m]
        counts = {'AB':0,'CD':0,'StringDB':0,'No Evidence':0}
        for i,row in rels.iterrows():
            tf = row['tf']
            gene = row['gene']
            if is_in_ABpkn(tf,gene):
                counts['AB'] += 1
            elif is_in_CDpkn(tf,gene):
                counts['CD'] += 1
            elif is_in_stringdb(tf,gene,all_networks):
                counts['StringDB'] += 1
            else:
                counts['No Evidence'] += 1
        models_counts['AB'].append(counts['AB'])
        models_counts['CD'].append(counts['CD'])
        models_counts['StringDB'].append(counts['StringDB'])
        models_counts['No Evidence'].append(counts['No Evidence'])

    models_counts = pd.DataFrame(models_counts)
    models_counts = (models_counts.T/models_counts.sum(axis=1)).T
    models_counts = models_counts.reset_index(names='size')
    models_counts['size'] += 1
    models_counts = models_counts.melt(id_vars=['size'],value_vars=['AB','CD','StringDB','No Evidence'],var_name='Confidence Group',value_name="Percentage of Relationships")
    print(models_counts)

    sns.lineplot(x='size',y='Percentage of Relationships',hue="Confidence Group",data=models_counts,markers=True,ax=ax)
    sns.scatterplot(x='size',y='Percentage of Relationships',hue="Confidence Group",data=models_counts,markers=True,ax=ax)

def get_model_num_filter_line_plot_noAB(ax):
    pairs_df = all_attr_pairs_unique
    models_counts = {'rand_CD':[],'CD':[],'StringDB':[],'No Evidence':[]}
    for m in range(1,max(list(pairs_df['size']))+1):
        rels = pairs_df[pairs_df['size'] >= m]
        counts = {'rand_CD':0,'CD':0,'StringDB':0,'No Evidence':0}
        for i,row in rels.iterrows():
            tf = row['tf']
            gene = row['gene']

            if is_in_randCDpkn(tf,gene):
                counts['CD'] += 1

            if is_in_CDpkn(tf,gene):
                counts['CD'] += 1
            elif is_in_stringdb(tf,gene,all_networks):
                counts['StringDB'] += 1
            else:
                counts['No Evidence'] += 1
        models_counts['rand_CD'].append(counts['rand_CD'])
        models_counts['CD'].append(counts['CD'])
        models_counts['StringDB'].append(counts['StringDB'])
        models_counts['No Evidence'].append(counts['No Evidence'])

    models_counts = pd.DataFrame(models_counts)
    #models_counts = (models_counts.T/models_counts.sum(axis=1)).T
    models_counts = models_counts.reset_index(names='Number of Models')
    models_counts['Number of Models'] += 1
    models_counts = models_counts.melt(id_vars=['Number of Models'],value_vars=['rand_CD','CD','StringDB','No Evidence'],var_name='Confidence Group',value_name="Number of Relationships")
    #models_counts = models_counts.melt(id_vars=['Number of Models'],value_vars=['rand_CD','CD','StringDB','No Evidence'],var_name='Confidence Group',value_name="Percentage of Relationships")
    print(models_counts)

    sns.lineplot(x='Number of Models',y='Number of Relationships',hue="Confidence Group",marker='o',data=models_counts,ax=ax)
    #sns.lineplot(x='Number of Models',y='Percentage of Relationships',hue="Confidence Group",marker='o',data=models_counts,ax=ax)
    #sns.scatterplot(x='Number of Models',y='Percentage of Relationships',hue="Confidence Group",data=models_counts,markers=True,ax=ax)


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

def is_in_randCDpkn(geneA,geneB):
    if geneA in set(rand_pknCD['source']):
        sourceCD = rand_pknCD[rand_pknCD['source'] == geneA]
        if geneB in set(sourceCD['target']):
            return True
    elif geneA in set(rand_pknCD['target']):
        sourceCD = rand_pknCD[rand_pknCD['target'] == geneA]
        if geneB in set(sourceCD['source']):
            return True
    return False

def is_in_ABpkn(geneA,geneB):
    if geneA in set(pknAB['source']):
        sourceAB = pknAB[pknAB['source'] == geneA]
        if geneB in set(sourceAB['target']):
            return True
    elif geneA in set(pknAB['target']):
        sourceAB = pknAB[pknAB['target'] == geneA]
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


def create_adjacency_df(pairs,network,ax,tick_labels=False):
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
    cluster = sns.clustermap(df)
    gene_ind = cluster.dendrogram_col.reordered_ind
    tf_ind = cluster.dendrogram_row.reordered_ind

    df = df.iloc[tf_ind,gene_ind]
    #clustergrid = sns.clustermap(df)
    #plt.clf()
    #df = df.iloc[clustergrid.dendrogram_row.reordered_ind,clustergrid.dendrogram_col.reordered_ind]
    #colors = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
    #colors = ["mediumpurple", "plum", "skyblue", "0.7", "0.9"]
    #colors = ["plum", "orange", "0.2", "0.9", "0.96"]
    colors = ["cornflowerblue", "orange", "0.2", "0.9", "0.96"]
    colors.reverse()
    my_cmap = ListedColormap(colors, name="my_cmap")
    heatmap(df,df.index,df.columns,ax=ax, norm=norm,cmap=my_cmap, cbar_kw=dict(ticks=np.arange(0,6),format=fmt),cbarlabel="Evidence Type",tick_labels=tick_labels)
    ax.set_ylabel('TF')
    ax.set_xlabel('Gene')
    return df, type_dict


def get_comb_heatmap(ax):
    df = create_adjacency_df(comb_pairs_df_filtered,all_networks,ax)
    print("done comb")

def generate_random_relationships(pkns,l):
    tfs = []
    genes = []
    for pkn in pkns:
        tfs = list(set(tfs + list(set(pkn['target']))))
        genes = list(set(genes + list(set(pkn['source']))))
    print(tfs)
    print(genes)
    tf_sample = np.random.choice(np.array(tfs),l)
    gene_sample = np.random.choice(np.array(genes),l)
    df = pd.DataFrame({'tf':tf_sample,'gene':gene_sample})
    print(df)
    df_unique = df.groupby(df.columns.tolist(),as_index=False).size()

    network = get_string_network_df(list(np.hstack([list(df_unique['gene']),list(df_unique['tf'])])))
    sleep(0.9)

    counts = {'AB':0,'CD':0,'StringDB':0,'No Evidence':0}
    for i,row in df_unique.iterrows():
        tf = row['tf']
        gene = row['gene']
        if is_in_ABpkn(tf,gene):
            counts['AB'] += 1
        elif is_in_CDpkn(tf,gene):
            counts['CD'] += 1
        elif is_in_stringdb(tf,gene,network):
            counts['StringDB'] += 1
        else:
            counts['No Evidence'] += 1

    return counts['CD']+counts['StringDB']

#print(generate_random_relationships([pknAB,pknCD],len(comb_pairs_df_filtered)))

#raise ValueError()

def get_heatmap(ax):
    #df = create_adjacency_df(all_attr_pairs_df,all_networks,ax,tick_labels=False)
    df, type_dict = create_adjacency_df(all_attr_pairs_unique,all_networks,ax,tick_labels=True)
    df[df < 2] = 0
    df[df > 4] = 0
    num_rels = np.count_nonzero(df)
    #rand_rels = np.array([generate_random_relationships([pknAB,pknCD],len(comb_pairs_df_filtered)) for i in range(50)])
    #pval = len(rand_rels[rand_rels >= num_rels])/len(rand_rels)
    #print(pval)
    print("done hm")

"""
#print("pval",len(random_dist[random_dist >= attr_dir_connections])/len(random_dist))
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
"""
