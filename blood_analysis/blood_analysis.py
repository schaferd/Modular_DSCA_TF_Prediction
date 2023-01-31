import numpy as np
import pandas as pd
import torch
import os
import sys
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

from pyensembl import EnsemblRelease
ensembl_data = EnsemblRelease(78)

encoder_path = os.environ["encoder_path"]
sys.path.insert(1,encoder_path)
from encoder import AEEncoder

is_gpu = False
device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
    is_gpu = True
print("is gpu 1"+str(is_gpu))

class BloodAnalysis():
    def __init__(self,ae_args):
        self.encoder = ae_args['encoder'].to(device)
        self.encoder.eval()
        self.tf_list = ae_args['tf_list']
        self.data_path = ae_args['data_path']
        self.celltype_path = ae_args['celltype_path']
        self.overlap_genes = ae_args['overlap_genes']
        self.ae_input_genes = ae_args['ae_input_genes']
        self.out_dir = ae_args['out_dir']
        self.input_df = self.get_input_df()
        self.input_data = self.format_data().to(device)
        self.activities = self.get_activities()
        self.labels = self.get_labels()
        self.train_log_reg()
        self.train_gmm()

    def get_input_df(self):
        data = pd.read_csv(self.data_path, delimiter="\t")
        data = data.set_index('GENE')
        data = ((data - data.mean())/data.std()).T
        new_cols = [self.convert_gene_name_to_ensembl(i)[0] for i in data.columns]
        new_cols_dict = {data.columns[i]:new_cols[i] for i in range(len(data.columns)) if new_cols[i]!= None}
        none_cols = [data.columns[i] for i in range(len(data.columns)) if new_cols[i] == None]
        data.drop(none_cols,axis=1,inplace=True)
        data.rename(columns=new_cols_dict,inplace=True)

        df0 = pd.DataFrame(0,index=[0],columns=self.ae_input_genes)

        input_df = pd.concat([df0,data],ignore_index=True)

        input_df = input_df.loc[1:,self.ae_input_genes]
        input_df = input_df.set_index(data.index)
        input_df = input_df.fillna(0)
        return input_df

    def format_data(self):
        matrix = torch.from_numpy(np.array(self.input_df).astype(np.float)).to(device).float()
        return matrix

    def get_activities(self):
        activities = self.encoder(self.input_data).cpu().detach().numpy()
        activities_df = pd.DataFrame(data=activities,index=self.input_df.index,columns=self.tf_list).sort_index()
        return activities_df

    def get_labels(self):
        return pd.read_csv(self.celltype_path,delimiter="\t").set_index("NAME").sort_index()

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
        clf = LogisticRegression(random_state=0,C=100).fit(tr_a,tr_l)
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
