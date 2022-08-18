import pandas as pd

import collections
import numpy as np
import random
import torch
from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)

class DataProcessing():

        def __init__(self,input_data,sparse_data,batch_size,relationships_filter):
                self.sparse_data_file = sparse_data.split("/")[-1]
                self.sparse_data = pd.read_csv(sparse_data, sep='\t', low_memory=False)
                self.sparse_data = self.sparse_data.dropna()

                self.batch_size = batch_size
                self.relationships_filter = relationships_filter

                self.tf_gene_dict = {}
                self.gene_tf_dict = {}

                self.input_data = pd.read_pickle(input_data)

                self.sparse_genes = np.unique(self.sparse_data.loc[:,'target'])
                """
                Convert sparse genes names to ensembl
                """
                new_sparse_genes = []
                gene_id_to_ensembl = {}
                for g in range(len(self.sparse_genes)):
                        try:
                                ensembl_id = ensembl_data.gene_ids_of_gene_name(self.sparse_genes[g])
                                new_sparse_genes.extend(ensembl_id)
                                gene_id_to_ensembl[self.sparse_genes[g]] = ensembl_id
                        except:
                                ensembl_id = None

                self.sparse_genes = np.array(new_sparse_genes)
                self.input_genes = self.input_data.columns

                self.gene_min, self.gene_max = self.get_data_min_max()

                self.overlapping_genes = set() 
                for g in self.sparse_genes:
                        if g in self.input_genes:
                            self.overlapping_genes.add(g)
                
                sparse_data_counter = 0
                for index,row in self.sparse_data.iterrows():
                        sparse_data_counter += 1
                        in_overlap_list = None
                        if row['target'] in gene_id_to_ensembl: 
                                ensembl_id = gene_id_to_ensembl[row['target']]
                                if ensembl_id is not None:
                                        for g in ensembl_id:
                                            if g in self.overlapping_genes:
                                                if g not in self.gene_tf_dict:
                                                    self.gene_tf_dict[g] = collections.OrderedDict() 
                                                self.gene_tf_dict[g][row['tf']] = row['mor']
                                                #self.gene_tf_dict[g][0].append(row['tf'])
                                                #self.gene_tf_dict[g][1].append(row['mor'])

                                                if row['tf'] not in self.tf_gene_dict:
                                                    self.tf_gene_dict[row['tf']] = collections.OrderedDict()

                                                self.tf_gene_dict[row['tf']][g] = row['mor']
                                                #self.tf_gene_dict[row['tf']][0].append(g)
                                                #self.tf_gene_dict[row['tf']][1].append(row['mor'])

                """
                filter tfs with gene sets < n 
                """
                genes_to_remove = set() 
                tfs_to_remove = set()
                for tf,vals in self.tf_gene_dict.items():
                    if len(vals) < self.relationships_filter or len(vals) > 300:
                        tfs_to_remove.add(tf)

                """
                for tf in self.tf_gene_dict.keys():
                    tfs_to_remove.add(tf)
                    if len(self.tf_gene_dict.keys())-len(tfs_to_remove) <10:
                        break
                """

                for gene,vals in self.gene_tf_dict.items():
                    is_valid = False
                    for tf in vals.keys():
                        if tf not in tfs_to_remove:
                            is_valid = True
                    if not is_valid:
                        genes_to_remove.add(gene)
                self.gene_tf_dict = {gene:val for gene,val in self.gene_tf_dict.items() if gene not in genes_to_remove}
                self.tf_gene_dict = {tf:val for tf,val in self.tf_gene_dict.items() if tf not in tfs_to_remove}
                self.overlapping_genes = set(self.gene_tf_dict.keys())

                
                

                #self.tfs = np.sort(np.unique(self.sparse_data.loc[:,'tf']))
                self.tfs = np.sort(np.unique(np.array(list(self.tf_gene_dict.keys()))))
                print("tfs",len(self.tfs))

                self.overlap_list = list(self.overlapping_genes)
                self.overlap_list.sort()
                print("overlap",len(self.overlap_list))

                self.input_data = self.input_data.loc[:,self.overlap_list]
                self.input_genes = self.input_data.columns
                self.genes = self.overlap_list

                #self.input_labels = self.input_data[self.overlap_list]
                self.labels = self.input_data.loc[:,self.overlap_list]
                self.gene_names = self.input_data.columns


        def get_data_min_max(self):
            #find min and max across genes
            gene_min = self.input_data.min(axis=0)
            gene_max = self.input_data.max(axis=0)
            return gene_min,gene_max

        def get_input_data(self):
                return self.input_data

        def get_output_data(self, input_data):
                return input_data.loc[:,self.overlap_list]

        def get_split_indices(self,dataset,k_splits):
            """
            Returns indices from dataset representing the start and end points for train
            and test sets for each fold in k-cross validation
            """
            rows = len(dataset.index)-100
            validation_split = (rows,rows+100)

            if k_splits == 1:
                splits = [(0,int(rows/2)),(int(rows/2),int(rows))]
                #splits = [(0,int(rows-5)),(int(rows-5),int(rows))]
                #splits = [(0,int(rows)),(0,int(rows))]
                return splits, [list(splits[0])], list(splits[1]), list(validation_split)

            rows_per_split = int(rows/k_splits)
            final_split = rows_per_split+(rows%k_splits)
            splits = [] #indices for each split
            curr_index = 0
            for split in range(k_splits):
                    if split == k_splits-1:
                            splits.append((curr_index,curr_index+final_split))
                    else:
                            splits.append((curr_index,curr_index+rows_per_split))
                            curr_index = int(curr_index + rows_per_split)
            train_splits = []
            test_splits = []
            for split in range(len(splits)):
                    test_splits.append(split)
                    train_split = [*range(k_splits)]
                    train_split.remove(split)
                    train_splits.append(train_split)
            return splits, train_splits, test_splits, validation_split


        def get_train_test_data(self,fold,k_splits,dataset,splits,train_splits,test_splits):
                    """
                    Returns train and test datasets for each fold for k-cross validation
                    """
                    train_data = None
                    test_data = None
                    if k_splits == 1:
                            train_data = dataset.iloc[train_splits[0][0]:train_splits[0][1]]
                            test_data = dataset.iloc[test_splits[0]:test_splits[1]]
                    else:
                            for split in range(k_splits):
                                    if split in train_splits[fold]: 
                                            train_split_indices = splits[split]
                                            if train_data is None:
                                                train_data = dataset.iloc[train_split_indices[0]:train_split_indices[1]]
                                            else:
                                                train_data = pd.concat([train_data,dataset[train_split_indices[0]:train_split_indices[1]]])
                                    if split == test_splits[fold]:
                                            test_split_indices = splits[split]
                                            if test_data is None:
                                                    test_data = dataset.iloc[test_split_indices[0]:test_split_indices[1]]
                                            else:
                                                    test_data = pd.concat([test_data,dataset[test_split_indices[0]:test_split_indices[1]]])
                    return train_data,test_data

        def convert_to_ensembl(self,g):
                ensembl_id = None
                try:
                        ensembl_id = ensembl_data.gene_ids_of_gene_name(self.sparse_genes[g])
                except:
                        ensembl_id = None
                return ensembl_id


