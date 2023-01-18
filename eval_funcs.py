import matplotlib.pyplot as plt
import seaborn as sns
import sys
import torch
import scipy
import numpy as np
import time
import os

from blood_analysis import BloodAnalysis

is_gpu = False
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    is_gpu = True
print("is gpu "+str(is_gpu))

roc_path = os.path.join(os.path.dirname(__file__),'ko_tests/diff_roc/')
print(roc_path)
sys.path.insert(1,roc_path)
from get_roc_curves import getROCCurve

roc_path = os.path.join(os.path.dirname(__file__),'essentiality/')
print(roc_path)
sys.path.insert(1,roc_path)
from get_roc_curves import getROCCurve as eROC

comp_path = os.path.join(os.path.dirname(__file__),'comp_dorothea/')
sys.path.append(comp_path)
from get_comp import getComp

is_gpu = False
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    is_gpu = True
print("is gpu "+str(is_gpu))

def get_correlation(model,data_loader):
    """
    run train data through final model then find correlation between input and output of final model
    """
    input_list = []
    output_list = []
    for samples, labels in data_loader:
        samples = samples.to(device)
        labels = labels.to(device)
        outputs = model(samples.float())
        input_list.append(labels.tolist())
        output_list.append(outputs.tolist())
        if len(input_list) > 5:
            break
    avg_corr,corr_list = correlation(input_list,output_list)
    return avg_corr,corr_list,input_list,output_list

def flatten_list(l):
        """
        Turns a 2D list l into a 1D list
        """
        new_list = []
        counter = 0
        for l_ in l:
                for item in l_:
                    new_list.append(item)
                    counter += 1
        return new_list

def correlation(ae_input,ae_output):
        avg_corr = 0
        #matrix where each row is a sample and each column is a gene
        full_matrix_input = None
        full_matrix_output = None
        for i in range(len(ae_input)):
                np_input = np.asarray(ae_input[i])
                np_output = np.asarray(ae_output[i])
                for j in range(np.shape(np_input)[0]):
                        if full_matrix_input is None:
                                full_matrix_input = np.asarray([np_input[j]])
                                full_matrix_output = np.asarray([np_input[j]])
                        else:
                                full_matrix_input = np.append(full_matrix_input,np.asarray([np_input[j]]),axis=0)
                                full_matrix_output = np.append(full_matrix_output,np.asarray([np_output[j]]),axis=0)

        #transpose the matrix so that each row is a gene
        full_matrix_input_T = (full_matrix_input+1e-8).T
        full_matrix_output_T = (full_matrix_output+1e-8).T
        corr_list = []
        scipy_start = time.time()
        #find correlation for each row/gene
        for gene in range(len(full_matrix_input_T)):
                corr = scipy.stats.pearsonr(full_matrix_input_T[gene],full_matrix_output_T[gene])[0]
                corr_list.append(corr)
        scipy_end = time.time()
        average_corr = sum(corr_list)/len(corr_list)
        return average_corr,corr_list

def get_correlation_between_runs(trained_models,data_loader,save_path):
        outputs = [] 
        for model in trained_models:
            for samples, labels in data_loader:
                output = model(samples.float())
                outputs.append(output.tolist())
        avg_pairwise_corr_list = []
        for i, output in enumerate(outputs):
                if i != len(outputs)-1:
                        remaining_outputs = outputs[i+1:]
                        for output2 in remaining_outputs:
                            pairwise_corr, corr_list = correlation([output],[output2]) 
                            avg_pairwise_corr_list.append(pairwise_corr)
        avg_corr = sum(avg_pairwise_corr_list)/len(avg_pairwise_corr_list)
        print("corr between runs: "+str(avg_corr))

        plt.clf()
        ax = sns.swarmplot(data=corr_list,color=".4",alpha=0.1)
        sns.boxplot(data=corr_list,ax=ax).set(title='corr btw runs')
        plt.savefig(save_path+'/corr_btw_runs_boxplot.png')
        plt.clf()

        return avg_corr, avg_pairwise_corr_list

def get_ko_roc_curve(data_obj,roc_data_path,encoder,save_path,fold=0,cycle=0):
    tf_gene_dict = {tf:data_obj.tf_gene_dict[tf].keys() for tf in data_obj.tf_gene_dict.keys()}
    print("GETTING KO ROC CURVE")
    print("data_obj.tfs",data_obj.tfs)
    print("overlap list",data_obj.overlap_list)
    ae_args = {
        'embedding':encoder,
        'overlap_genes': data_obj.overlap_list,
        'knowledge':tf_gene_dict,
        'data_dir':roc_data_path,
        'ae_input_genes':data_obj.input_genes,
        'tf_list':data_obj.tfs,
        'out_dir':save_path,
        'fold':fold,
        'cycle':cycle
    }
    obj = getROCCurve(ae_args=ae_args)
    return obj.auc, obj.diff_activities, obj.scaled_rankings
"""
def get_essentiality_roc_curve(data_obj,roc_data_path,encoder,save_path,fold=0,cycle=0):
    tf_gene_dict = {tf:data_obj.tf_gene_dict[tf].keys() for tf in data_obj.tf_gene_dict.keys()}
    ae_args = {
        'embedding':encoder,
        'overlap_genes': data_obj.overlap_list,
        'knowledge':tf_gene_dict,
        'data_dir':roc_data_path,
        'ae_input_genes':data_obj.input_genes,
        'tf_list':data_obj.tfs,
        'out_dir':save_path,
        'fold':fold,
        'cycle':cycle
    }
    obj = eROC(ae_args=ae_args)
    return obj.auc, obj.diff_activities, obj.scaled_rankings

def comp_dorothea(data_obj,roc_data_path,encoder,save_path,fold=0,cycle=0):
    tf_gene_dict = {tf:data_obj.tf_gene_dict[tf].keys() for tf in data_obj.tf_gene_dict.keys()}
    ae_args = {
        'embedding':encoder,
        'overlap_genes': data_obj.overlap_list,
        'knowledge':tf_gene_dict,
        'data_dir':roc_data_path,
        'ae_input_genes':data_obj.input_genes,
        'tf_list':data_obj.tfs,
        'out_dir':save_path,
        'fold':fold,
        'cycle':cycle
    }
    obj = getComp(ae_args=ae_args)
    return obj
"""

def get_blood_analysis(data_obj, blood_data_path, celltype_path, save_path, encoder,fold=0,cycle=0):
    ae_args = {
        'encoder':encoder,
        'overlap_genes': data_obj.overlap_list,
        'celltype_path':celltype_path,
        'data_path':blood_data_path,
        'ae_input_genes':data_obj.input_genes,
        'tf_list':data_obj.tfs,
        'out_dir':save_path,
        'fold':fold,
        'cycle':cycle
    }
    obj = BloodAnalysis(ae_args=ae_args)






