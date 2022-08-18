import argparse
import sparselinear
import gc
from argparse import RawTextHelpFormatter
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import subprocess
from torch.utils.checkpoint import checkpoint_sequential
import sys
import time
import random
import torch
import pandas as pd
from torch import nn
import torch
from collections import OrderedDict
from moa import MOA
import shutil

import pickle as pkl
from data_class import CellTypeDataset
from ae_model import AE
from eval_funcs import get_correlation,get_correlation_between_runs,get_roc_curve
from figures import plot_input_vs_output,create_test_vs_train_plot,create_corr_hist,create_moa_figs

from data_processing import DataProcessing

#device = torch.device('cpu')
is_gpu = False
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    is_gpu = True
print("is gpu "+str(is_gpu))

encoder_path = os.environ["encoder_path"]
sys.path.insert(1,encoder_path)
from encoder import AEEncoder

decoder_path = os.environ["decoder_path"]
sys.path.insert(1,decoder_path)
from decoder import AEDecoder

class Train():

    def __init__(self,param_dict):

        self.param_dict = param_dict
        self.loss = None

        self.input_path = self.param_dict["input_path"]
        self.sparse_path = self.param_dict["sparse_path"]
        self.base_path = self.param_dict["save_base_path"]

        self.save_figs = self.param_dict["save_figs"]
        self.save_model = self.param_dict["save_model"]
        self.fig_freq = self.param_dict["fig_freq"]
        self.model_type = self.param_dict["model_type"]

        self.record = self.param_dict["record"]
        self.record_path = self.param_dict["record_path"]

        self.epochs = self.param_dict["epochs"]
        self.batch_size = self.param_dict["batch_size"]
        self.lr = self.param_dict["lr"]
        self.lr_sched = self.param_dict["lr_sched"]
        self.max_lr = self.param_dict["max_lr"]
        self.warm_restart = self.param_dict["warm_restart"]
        self.width_multiplier = self.param_dict["width_multiplier"]

        self.l2_reg = self.param_dict["l2_reg"]
        self.l1 = self.param_dict["l1"]
        self.dropout_rate = self.param_dict["dropout_rate"]
        self.batch_norm = self.param_dict["batch_norm"]
        self.relationships_filter = self.param_dict["relationships_filter"]

        self.moa_beta = self.param_dict["moa_beta"]
        self.moa_subset = self.param_dict["moa_subset"]
        self.moa = self.param_dict["moa"]

        self.roc_data_path = self.param_dict["roc_data_path"]

        self.data_obj = DataProcessing(self.input_path,self.sparse_path,self.batch_size,self.relationships_filter)
        self.MOA = MOA(self.data_obj,self.moa,self.moa_subset,self.moa_beta)
        
        self.encoder = AEEncoder(data=self.data_obj,dropout_rate=self.dropout_rate,batch_norm=self.batch_norm,width_multiplier=self.width_multiplier).to(device)
        self.decoder = AEDecoder(data=self.data_obj,dropout_rate=self.dropout_rate,batch_norm=self.batch_norm,width_multiplier=self.width_multiplier).to(device)
        self.model = AE(self.encoder,self.decoder)

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.l2_reg)
        self.criterion = nn.MSELoss()

        self.save_path = None 
        shutil.copyfile(encoder_path+'/encoder.py',self.get_save_path()+'/encoder.py')
        shutil.copyfile(decoder_path+'/decoder.py',self.get_save_path()+'/decoder.py')

        self.scheduler = None 

        self.warm_restart_freq = 0
        if self.warm_restart > 0:
            self.warm_restart_freq = self.epochs//(self.warm_restart+1)

        self.train_output = None
        self.test_output = None
        self.train_error = None
        self.test_error = None

        self.training_losses = [[]]
        self.test_losses = [[]]

        self.moa_train_losses = []
        self.moa_test_losses = []
        self.moa_violation_count_test = []
        self.moa_violation_count_train = []
        self.embedding_losses = []
        self.trained_models = []

        self.trained_embedding_model = None

        
        if self.save_figs:
                param_file = open(self.get_save_path()+"/params",'w+')
                param_file.write(repr(self.param_dict))
                param_file.close()


    def get_lowest_loss(self,fold):
        """
        Sets self.loss to last element of list if last element is less than self.loss
        Returns self.loss
        """
        if (self.loss is None and len(self.test_losses[fold]) > 0) or (self.loss is not None and len(self.test_losses[fold]) > 0 and self.test_losses[fold][-1] < self.loss):
                self.loss = self.test_losses[fold][-1]
        return self.loss

    

    def init_weights(self,m):
        """
        Initilizes weights in network m
        """
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.1)


    def get_trained_model(self,train_loader,test_loader,fold_num=0,):
        """
        Trains and returns train loss, test loss and correlation
        saves entire model, encoder model, figures and correlation if specified
        """

        if self.save_figs:
            print("saving at "+str(self.get_save_path()))
        else:
            print("NOT saving at "+str(self.get_save_path()))

        self.model.encoder.apply(self.init_weights)
        self.model.decoder.apply(self.init_weights)
        test_loss = None
        train_loss = None

        train_start = time.time()

        for epoch in range(self.epochs):
            if self.warm_restart != 0 and epoch%self.warm_restart_freq == 0:
                if self.lr_sched:
                    sched_state = self.scheduler.state_dict()
                    self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.max_lr,epochs=self.epochs,steps_per_epoch=len(train_loader))
                    self.scheduler.load_state_dict(sched_state)
                self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.l2_reg)

            train_loss = self.train_iteration(train_loader,fold_num)
            test_loss = self.test_iteration(test_loader,fold_num)
            print("epoch : {}/{}, test loss = {:6f}".format(epoch+1,self.epochs,test_loss))
            print("epoch : {}/{}, train loss = {:6f}".format(epoch+1,self.epochs,train_loss))
            if self.save_figs and epoch%self.fig_freq == 0 and epoch > 0:
                self.save_train_test_loss_data()
                #if self.save_model:
                #    torch.save(self.model.module.encoder,self.get_save_path()+"/model_encoder_fold"+str(fold_num)+"_epoch_"+str(epoch)+".pth")
                if self.moa > 0:
                        create_moa_figs(self.moa_train_losses,self.moa_test_losses,self.moa_violation_count_train,self.moa_violation_count_test,self.get_save_path(),fold=fold_num)
                create_test_vs_train_plot(self.training_losses,self.test_losses,self.get_save_path(),fold_num)
        
        
        self.trained_embedding_model = self.model.encoder
        self.trained_models.append(self.model)

        
        correlation, corr_list,input_list,output_list = get_correlation(self.model,train_loader)
        test_correlation, test_corr_list,test_input_list,test_output_list = get_correlation(self.model,test_loader)
        print("correlation "+str(correlation))

        print("test correlation "+str(correlation))

        auc = 0
        if self.save_model:
            torch.save(self.model.encoder,self.get_save_path()+"/model_encoder_fold"+str(fold_num)+".pth")
            corr_file = open(self.get_save_path()+"/corr_fold"+str(fold_num),'w+')
            corr_file.write(str(correlation)+"\n")
            corr_file.write(str(corr_list))
            corr_file.close()
        if self.save_figs:
            self.save_train_test_loss_data()
            if self.moa > 0:
                    create_moa_figs(self.moa_train_losses,self.moa_test_losses,self.moa_violation_count_train,self.moa_violation_count_test,self.get_save_path(),fold=fold_num)
            create_test_vs_train_plot(self.training_losses,self.test_losses,self.get_save_path(),fold_num)
            plot_input_vs_output(input_list,output_list,correlation,False,self.get_save_path(),fold=fold_num)
            plot_input_vs_output(test_input_list,test_output_list,test_correlation,True,self.get_save_path(),fold=fold_num)
            create_corr_hist(corr_list,correlation,self.get_save_path(),False,fold=fold_num)
            create_corr_hist(test_corr_list,test_correlation,self.get_save_path(),True,fold=fold_num)
            auc = get_roc_curve(self.data_obj,self.roc_data_path,self.trained_embedding_model,self.get_save_path(),fold=fold_num)
        if self.moa > 0:
                self.moa_test_losses = []
                self.moa_train_losses = []
        if self.record:
            self.add_to_record(auc,train_loss,test_loss,correlation,test_correlation)

        return train_loss,test_loss,correlation



    def train_iteration(self,train_loader,fold):

        """
        Completes one training iteration
        """
        loss = 0
        epoch_moa_loss = 0
        epoch_moa_violations = 0
        self.model.train()
        for samples, labels in train_loader:

            samples = samples.to(device)

            self.optimizer.zero_grad(set_to_none=True)
            
            outputs = self.model(samples.float())
            self.train_output = outputs
            train_loss = self.criterion(outputs.to(device),labels.float().to(device))

            moa_loss = 0
            if self.moa > 0: 
                moa_loss,violations = self.MOA.get_moa_loss(self.model.decoder)
                epoch_moa_loss += moa_loss.cpu().detach()
                epoch_moa_violations += violations.cpu().detach()

            train_loss = train_loss+moa_loss
            
            """
            if len(self.training_losses)<(fold+1):
                    self.training_losses.append([train_loss])
            else:
                    self.training_losses[fold].append(train_loss)
            """
            train_loss.backward()
            self.optimizer.step()
            loss += train_loss.detach().item()
            if self.scheduler is not None:
                self.scheduler.step()
                
        if self.moa > 0:
                epoch_moa_loss = epoch_moa_loss/len(train_loader)
                epoch_moa_violations = epoch_moa_violations/len(train_loader)
                self.moa_train_losses.append(epoch_moa_loss.detach())
                self.moa_violation_count_train.append(epoch_moa_violations.detach())

        loss = loss/len(train_loader)
        if len(self.training_losses)<(fold+1):
                self.training_losses.append([loss])
        else:
                self.training_losses[fold].append(loss)

        print("train loss = "+ str(loss))
        return loss


    def test_iteration(self,test_loader,fold):
        """
        Performs a test iteration 
        Returns loss
        """
        loss = 0
        epoch_moa_loss = 0
        epoch_moa_violations = 0
        self.model.eval()
        for samples, labels in test_loader:
                samples = samples.to(device)
                labels = labels.to(device)
                self.model.to(device)
                test_output = self.model(samples.float())
                self.test_output = test_output
                test_loss = self.criterion(test_output,labels.float())

                moa_loss=0
                if self.moa > 0:
                    moa_loss,violations = self.MOA.get_moa_loss(self.model.decoder)
                    epoch_moa_loss += moa_loss.cpu().detach()
                    epoch_moa_violations += violations.cpu().detach()
                test_loss = test_loss+moa_loss
                loss += test_loss.item()

        if self.moa > 0:
                epoch_moa_loss = epoch_moa_loss/len(test_loader)
                epoch_moa_violations = epoch_moa_violations/len(test_loader)
                self.moa_test_losses.append(epoch_moa_loss.detach())
                self.moa_violation_count_test.append(epoch_moa_violations.detach())

        loss = loss/len(test_loader)
        if len(self.test_losses)<(fold+1):
                self.test_losses.append([loss])
        else:
                self.test_losses[fold].append(loss)
        print("test loss = "+ str(loss))
        self.get_lowest_loss(fold)
        return loss


    def save_train_test_loss_data(self):
        """
        Saves a table of train and test losses for each epoch
        """
        with open(self.get_save_path()+'/rmse', 'w+') as f:
            f.write("epoch  train  test \n")
            for i in range(len(self.training_losses)):
                f.write(str(i)+" "+str(self.training_losses[i])+"\n "+str(self.test_losses[i])+"\n")

    def get_save_path(self):
        """
        returns a unique save directory for the run
        """
        if self.save_path is not None:
            return self.save_path 

        model_type = ''
        if self.model_type != 'none':
            model_type = self.model_type+'_'

        time_tuple = time.localtime(time.time())
        time_for_save = str(time_tuple[1])+"-"+str(time_tuple[2])+"_"+str(time_tuple[3])+"."+str(time_tuple[4])+"."+str(time_tuple[5])

        encoder_name = encoder_path.rstrip('/').split('/')[-1]
        decoder_name = decoder_path.rstrip('/').split('/')[-1]

        if encoder_name == 'gene_grouped_fc_indep':
            encoder_name = 'genefc'
        elif encoder_name == 'gene_grouped_indep':
            encoder_name = 'gene'
        elif encoder_name == 'tf_grouped_fc_indep':
            encoder_name = 'tffc'
        elif encoder_name == 'tf_grouped_indep':
            encoder_name = 'tf'

        if decoder_name == 'gene_grouped_fc_indep':
            decoder_name = 'genefc'
        elif decoder_name == 'gene_grouped_indep':
            decoder_name = 'gene'
        elif decoder_name == 'tf_grouped_fc_indep':
            decoder_name = 'tffc'
        elif decoder_name == 'tf_grouped_indep':
            decoder_name = 'tf'

        save_path = self.base_path+str(model_type)+encoder_name+'-'+decoder_name+"_epochs"+str(self.epochs)+"_batchsize"+str(self.batch_size)+"_lr"+str(self.lr)

        if self.lr_sched:
            save_path += "_lrsched"
        if self.l2_reg > 0:
            save_path += "_l2"+str(self.l2_reg)
        if self.l1 > 0:
            save_path += "_l1"+str(self.l1)
        if self.moa > 0:
            save_path +="_moa"+str(self.moa)
        if self.dropout_rate > 0:
            save_path += "_do"+str(self.dropout_rate)
        if self.warm_restart > 0:
            save_path += "_wr"+str(self.warm_restart)
        if self.batch_norm:
            save_path += "_batchnorm"
        if self.relationships_filter > 1:
            save_path += "_rel_conn"+str(self.relationships_filter)

        save_path +="_"+str(time_for_save)

        self.save_path = save_path 

        if self.save_figs:
                os.mkdir(save_path)

        return self.save_path


    def add_to_record(self,auc,train_rmse,test_rmse,train_corr,test_corr):
        time_tuple = time.localtime(time.time())
        time_for_save = str(time_tuple[1])+"-"+str(time_tuple[2])+"_"+str(time_tuple[3])+"."+str(time_tuple[4])+"."+str(time_tuple[5])

        save_param_dict = self.param_dict.copy()
        del save_param_dict['fig_freq']
        save_param_dict['run_type'] = save_param_dict["model_type"]
        save_param_dict['sparse_path'] = save_param_dict['sparse_path'].split('/')[-1]
        save_param_dict['model_type'] = save_param_dict['save_base_path'].split('/')[-1]
        del save_param_dict['save_base_path']

        save_param_dict['ROC_AUC'] = auc
        save_param_dict['test corr'] = test_corr
        save_param_dict['train corr'] = train_corr
        save_param_dict['train_rmse'] = train_rmse
        save_param_dict['test_rmse'] = test_rmse
        save_param_dict['save_path'] = self.get_save_path()
        save_param_dict['time'] = time_for_save

        print("save param dict",save_param_dict)

        save_param_dict = {key:[save_param_dict[key]] for key in save_param_dict.keys()}

        try:
            data = pd.read_pickle(self.record_path)
        except:
            data = pd.DataFrame(columns=save_param_dict.keys())
        record_csv = '/'.join(self.record_path.split('/')[:-1])+'/'+self.record_path.split('/')[-1].split('.')[0]+".csv"
        print(record_csv)

        new_row = pd.DataFrame.from_dict(save_param_dict)
        new_df = pd.concat([data,new_row],ignore_index=True)

        new_df.to_csv(record_csv)
        new_df.to_pickle(self.record_path)

    def cross_validation(self,k_splits):
        dataset = self.data_obj.get_input_data()
        splits, train_splits, test_splits, validation_split = self.data_obj.get_split_indices(dataset,k_splits) 
        dataset = dataset.sample(frac=1)
        for fold in range(k_splits): 
                prep_data_start = time.time()
                print('Fold {}'.format(fold + 1))
                train_data, test_data = self.data_obj.get_train_test_data(fold,k_splits,dataset,splits,train_splits,test_splits)
                train_data_celltype = CellTypeDataset(train_data,self.data_obj.get_output_data(train_data))
                test_data_celltype = CellTypeDataset(test_data,self.data_obj.get_output_data(test_data))

                kwargs = {'num_workers':0,'pin_memory':True} if is_gpu == True else {}

                train_loader = torch.utils.data.DataLoader(train_data_celltype, batch_size=batch_size, **kwargs)
                test_loader = torch.utils.data.DataLoader(test_data_celltype, batch_size=batch_size, **kwargs)


                if self.lr_sched:
                    self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.max_lr,epochs=self.epochs,steps_per_epoch=len(train_loader))

                loss_list = []
                corr_list = []
                print("prep data time "+str(time.time()-prep_data_start))
                get_trained_model_start = time.time()
                train_loss,test_loss, corr = self.get_trained_model(train_loader,test_loader,fold)
                print("get trained model time "+str(time.time()-get_trained_model_start))
                loss_list.append(test_loss)
                corr_list.append(corr)

        if k_splits > 1:
                validation_data = dataset.iloc[validation_split[0]:validation_split[1]]
                validation_data_celltype = CellTypeDataset(validation_data,self.data_obj.get_output_data(validation_data))
                validation_loader = torch.utils.data.DataLoader(validation_data_celltype,batch_size=batch_size)
                get_correlation_between_runs(self.trained_models,validation_loader,self.get_saved_path())



if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='a cool autoencoder!!',formatter_class=RawTextHelpFormatter)
        parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train on')
        parser.add_argument('--learning_rate', type=float, required=False,default=1e-2, help='the learning rate')
        parser.add_argument('--save_figs',type=str,required=False,default=False,help='set to True if you want figures from training/testing to be saved')
        parser.add_argument('--save_path',type=str,required=False,default="figures/",help='directory where you want error data and figures to saved')
        parser.add_argument('--fig_freq',type=int,required=False,default=10,help='how often do you want figures on epochs (ie. every 10 epochs)')
        parser.add_argument('--model_type',type=str,required=False,default='ae',help='describe model')
        parser.add_argument('--l2',type=float,required=False,default=0,help='value for l2 regularization')
        parser.add_argument('--dropout',type=float,required=False,default=0,help='prob of a given node being deactivated')
        parser.add_argument('--save_model',type=str,required=False,default=False,help='whether you want the model data to be saved')
        parser.add_argument('--input_data_path',type=str,required=True,help='path for input data')
        parser.add_argument('--sparse_data_path',type=str,required=True,help='path for sparse data')
        parser.add_argument('--lr_sched',type=str,required=False,default=False,help='True if you want learning rate to be scheduled')
        parser.add_argument('--batch_norm',type=str,required=False,default=False,help='True if you want batch normalization layers')
        parser.add_argument('--batch_size',type=int,required=True,help='size of training batches')
        parser.add_argument('--l1',type=float,required=False,default=0,help='value for l1 reg')
        parser.add_argument('--warm_restart',type=int,required=False,default=0,help='how many times during training do you want adam to restart')
        parser.add_argument('--max_lr',type=float,required=False,default=1e-3,help='max value lr_sched will reach')
        parser.add_argument('--k_splits',type=int,required=False,default=1,help='how many splits you want in cross validation')

        parser.add_argument('--width_multiplier',type=int,required=False,default=1,help="width of network = width_multiplier*input_size")
        parser.add_argument('--relationships_filter',type=int,required=False,default=1,help='minimum connections that genes must have to be in output layer')

        parser.add_argument('--moa',type=float,required=False,default=0,help='value to multiple moa loss by')
        parser.add_argument('--moa_beta',type=float,required=False,default=0.9,help='beta value for moa')
        parser.add_argument('--moa_subset',type=int,required=False,default=0,help='subset value for moa')

        parser.add_argument('--roc_data_path',type=str,required=True,help='path to roc data')

        parser.add_argument('--record',type=str,required=False,default=False,help="true if you want results to recorded in record table")
        parser.add_argument('--record_path',type=str,required=True,help="where you want to keep the record/where record is kept")

        args = parser.parse_args()

        epochs = args.epochs 
        batch_size = args.batch_size

        fig_freq = args.fig_freq
        model_type = args.model_type
        width_multiplier = args.width_multiplier
        relationships_filter = args.relationships_filter

        l2_reg = args.l2
        l1 = args.l1
        batch_norm = args.batch_norm.lower() == 'true'
        dropout_rate = args.dropout
        k = args.k_splits
        
        sparse_path = args.sparse_data_path
        save_figs = args.save_figs.lower() == 'true'
        save_model = args.save_model.lower() == 'true'
        input_path = args.input_data_path


        lr = args.learning_rate
        lr_sched = args.lr_sched.lower()=='true'
        max_lr = args.max_lr
        warm_restart = args.warm_restart

        moa_subset = args.moa_subset
        moa_beta = args.moa_beta
        moa = args.moa

        roc_data_path = args.roc_data_path

        record = args.record.lower() == 'true'
        record_path = args.record_path.lower()

        print(args)	

        hyper_params = {
            "epochs":epochs,
            "batch_size":batch_size,
            "fig_freq":fig_freq,
            "model_type":model_type,	
            "width_multiplier":width_multiplier,

            "lr":lr,
            "lr_sched":lr_sched,
            "max_lr":max_lr,
            "warm_restart":warm_restart,

            "l2_reg":l2_reg,
            "l1":l1,
            "batch_norm":batch_norm,
            "dropout_rate":dropout_rate,

            "input_path":input_path,
            "sparse_path":sparse_path,
            "save_figs":save_figs,
            "save_model":save_model,
            "save_base_path":args.save_path,
            "relationships_filter":relationships_filter,

            "moa":moa,
            "moa_subset":moa_subset,
            "moa_beta":moa_beta,

            "roc_data_path":roc_data_path,

            "record":record,
            "record_path":record_path
        }
        train = Train(hyper_params)

        entire_train_process_start = time.time()
        train.cross_validation(k)
        print("entire train process "+str(time.time()-entire_train_process_start))


