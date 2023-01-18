import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)

class BloodAnalysis():
    def __init__(self,activities, metadata, out_dir):
        self.activities = pd.read_csv(activities).set_index("Unnamed: 0").T
        self.labels = self.get_labels(metadata)
        print("activities",self.activities)
        print("metadata",self.labels)
        self.out_dir = out_dir
        self.train_log_reg()
        self.train_gmm()
        

    def get_labels(self,metadata_path):
        return pd.read_csv(metadata_path ,delimiter="\t").set_index("NAME").sort_index()

    def get_train_test_data(self,train_size=700):
        self.activities = self.activities[self.activities.index.isin(self.labels.index)]
        self.labels = self.labels[self.labels.index.isin(self.activities.index)]

        train_labels = self.labels.sample(n=train_size)
        train_activities = self.activities[self.activities.index.isin(train_labels.index)]

        test_labels = self.labels[~self.labels.index.isin(train_labels.index)]
        test_activities = self.activities[self.activities.index.isin(test_labels.index)]

        return train_labels.to_numpy().T.flatten(), train_activities.to_numpy(), test_labels.to_numpy().T.flatten(), test_activities.to_numpy()

    def train_log_reg(self):
        tr_l, tr_a, t_l, t_a = self.get_train_test_data()
        clf = LogisticRegression(random_state=0).fit(tr_a,tr_l)
        pred = clf.predict(t_a)
        nmis = normalized_mutual_info_score(t_l,pred)
        print("logistic regression nmis",nmis)
        accuracy = clf.score(t_a,t_l)
        print("logistic regression accuracy", accuracy)
        return nmis, accuracy

    def train_gmm(self):
        tr_l, tr_a, t_l, t_a = self.get_train_test_data()
        gm = GaussianMixture(n_components=3,random_state=0,n_init=5).fit(tr_a,tr_l)
        pred = gm.predict(t_a)
        nmis = normalized_mutual_info_score(t_l,pred)
        print("gm regression nmis",nmis)
        accuracy = gm.score(t_a,t_l)
        print("gm accuracy",accuracy)
        return nmis, accuracy

    def convert_gene_name_to_ensembl(self,gene_name):
        try:
            ensembl_id = ensembl_data.gene_ids_of_gene_name(gene_name)
            return ensembl_id
        except:
            return [None]


if __name__ == "__main__":
    dorothea_activities_path = "/nobackup/users/schaferd/blood_analysis_data/SCP43/expression/blood_data.viper_pred.csv"
    metadata_path = "/nobackup/users/schaferd/blood_analysis_data/SCP43/metadata/metadata.txt"
    outdir = "/nobackup/users/schaferd/blood_analysis_data/SCP43/expression/"
    obj = BloodAnalysis(dorothea_activities_path,metadata_path,outdir)
