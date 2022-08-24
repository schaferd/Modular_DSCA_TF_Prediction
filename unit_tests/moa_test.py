import numpy as np
import scipy
import sys
import torch
import os
import matplotlib.pyplot as plt
from torch import nn
import seaborn as sns
from datetime import datetime



train_path = '/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/'
sys.path.insert(1,train_path)
from moa import MOA
from data_processing import DataProcessing

decoder_path = train_path+'gene_grouped_indep/'
sys.path.insert(1,decoder_path)
from decoder import AEDecoder 

sys.setrecursionlimit(100000)
sns.color_palette("Blues", as_cmap=True)

is_gpu = False
if torch.cuda.is_available():
    device = torch.device('cuda')
    is_gpu = True
print("is gpu "+str(is_gpu))

criterion = nn.MSELoss()
now = datetime.now()


class UnitTest():
        def __init__(self,decoder,data_obj,moa_factor,trials,epochs,outpath):
            self.decoder = decoder
            self.moa_factor = moa_factor
            self.trials = trials
            self.epochs = epochs
            self.data_obj = data_obj
            self.outpath = outpath
            self.subset = 0
            self.beta = 0.9

            self.moa = MOA(self.data_obj,self.moa_factor,self.subset,self.beta)

            date_str = now.strftime("%m_%d_%Y__%H_%M")
            self.outpath = outpath+"/trials"+str(trials)+"_epochs"+str(epochs)+"_factor"+str(moa_factor)+"_"+date_str+"/"
            print("saving at")
            print(self.outpath)
            if not os.path.isdir(self.outpath):
                os.mkdir(self.outpath)

            #self.get_runs()
            self.unit_test(1)


        def get_runs(self):
            violation_matrices = []
            for i in range(self.trials):
                violation_matrices.append(self.unit_test(i))

            self.get_freq_violations(violation_matrices)

        def get_freq_violations(self,violation_matrices):
            violation_matrices = [matrix.type(torch.int64) for matrix in violation_matrices]
            violation_matrices = torch.stack(violation_matrices)
            violation_freq = torch.sum(violation_matrices,0).cpu().detach().numpy()

            nzs = np.transpose(np.nonzero(violation_freq))
            val_list = [violation_freq[coord[0]][coord[1]] for coord in nzs]
            plt.hist(val_list)
            plt.savefig(self.outpath+"freq_hist.png")

            print("avg val of violation")
            viol_list = [self.moa_matrix[coord[0]][coord[1]] for coord in nzs]
            print(sum(viol_list)/len(viol_list))
            
        def unit_test(self,trial):
            decoder.train()
            optimizer = torch.optim.Adam(decoder.parameters(),lr=0.01,weight_decay=0)
            decoder.apply(self.init_weights)
            input_tensor = torch.rand(len(data_obj.tfs)).to(device)
            output_tensor = torch.rand(len(data_obj.overlap_list)).to(device)
            loss_list = []
            moa_loss_list = []
            violations_list = []
            curr_violation_matrix = None
            for i in range(epochs):
                #print("param data")
                #for name, param in self.decoder.named_parameters():
                #    print(name, param.data)
                #w1 = list(self.decoder.parameters())[0]
                #b1 = np.expand_dims(list(self.decoder.parameters())[1].cpu().detach().numpy(),axis=0)
                #row = len(self.data_obj.tfs)
                #col = len(self.data_obj.overlap_list)
                #w1 = scipy.sparse.coo_matrix((w1.cpu().detach().numpy(),(self.data_obj.original_sparse_tensor[1],self.data_obj.original_sparse_tensor[0])),shape=(row,col))
                #print(w1.shape,b1.shape)
                #np_probe = np.concatenate((np.identity(row),np.zeros((1,row))),axis=0)
                #print((np_probe@w1)+b1)

                optimizer.zero_grad()
                output = decoder(input_tensor)
                loss = criterion(output,output_tensor)

                moa_loss,violations = self.moa.get_moa_loss(decoder)

                loss = loss+moa_loss
                loss_list.append(loss.cpu().detach().item())
                moa_loss_list.append(moa_loss.cpu().detach().item())
                violations_list.append(violations.cpu().detach().item())

                loss.backward()
                #print(decoder.weights.grad)
                #make sure not none or matrix of zeros
                optimizer.step()

            self.create_moa_figs(moa_loss_list,violations_list,self.outpath,trial)
            self.create_test_vs_train_plot(loss_list,self.outpath,trial)

        def init_weights(self,m):
            """
            Initilizes weights in network m
            """
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.1)

        def create_moa_figs(self,moa_losses,moa_violation_count,save_path,fold=0):
            fig = plt.figure(5)
            plt.plot(np.array(moa_losses),label="train loss")
            plt.title("MOA Loss fold"+str(fold))
            plt.xlabel('epochs')
            plt.ylabel('moa loss')
            plt.savefig(save_path+'/moa_loss_fold'+str(fold)+".png")
            plt.clf()

            fig = plt.figure(5)
            plt.plot(np.array(moa_violation_count),label="train violations")
            plt.title("MOA Violations fold"+str(fold))
            plt.xlabel('epochs')
            plt.ylabel('num violations')
            plt.legend()
            plt.savefig(save_path+'/moa_violations_fold'+str(fold)+".png")
            plt.clf()

        def create_test_vs_train_plot(self,losses,save_path,fold=0):
            """
            Saves RMSE vs. Epochs line plot
            """
            fig = plt.figure(4)
            training_losses = [tensor for tensor in losses]# if type(tensor) == torch.Tensor]
            tr_rmse = training_losses[-1]

            plt.plot(np.array(training_losses),label="Train")
            plt.xlabel('Epochs')
            plt.ylabel('RMSE')
            plt.title("Train RMSE "+str(round(tr_rmse,2)))
            plt.ylim(bottom=0)
            plt.xlim(left=0)
            plt.legend()
            plt.savefig(save_path+'/rmse_fold'+str(fold)+'.png')
            plt.clf()



if __name__ == "__main__":
    is_hdf_file = True
    relationships_filter = 1
    batch_size = 128
    train_path = "/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/gene_grouped_indep/"
    sparse_data = "/nobackup/users/schaferd/ae_project_data/dorothea_tf_gene_relationship_knowledge/dorotheaSelectionA.tsv"
    input_data = "/nobackup/users/schaferd/ae_project_data/hdf_gene_expression_data/agg_data.pkl"
    data_obj = DataProcessing(input_data,sparse_data,batch_size,relationships_filter)
    decoder = AEDecoder(data=data_obj,dropout_rate=0,batch_norm=False,width_multiplier=2).to(device)
    moa_factor = 1
    trials = 1 
    epochs = 20
    outpath = "/nobackup/users/schaferd/ae_project_outputs/unit_tests/moa/"
    test = UnitTest(decoder,data_obj,moa_factor,trials,epochs,outpath)
