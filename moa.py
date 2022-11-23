import torch
import numpy as np
is_gpu = False
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    is_gpu = True
print("is gpu "+str(is_gpu))

class MOA():
    def __init__(self,data_obj,moa_factor,subset,beta):
        self.data_obj = data_obj
        tf_probing = (torch.eye(len(self.data_obj.tfs))).to(device)
        control_row = torch.zeros((1,len(self.data_obj.tfs))).to(device)
        self.probe = torch.cat((tf_probing,control_row),axis=0).to(device)
        self.moa_matrix = self.get_tf_gene_matrix()
        self.moa_factor = moa_factor
        self.subset = subset
        self.beta = beta

    def get_moa_loss(self,decoder):
        #create probe tensor
        moa_matrix = self.moa_matrix
        decoder_oputput = None
        mask = None
        probe = self.probe
        if self.subset != 0:
            selectedIndex = np.arange(len(self.data_obj.tfs))
            selectedIndex = np.random.permutation(selectedIndex)[0:self.subset]
            TF_index = selectedIndex.copy()
            selectedIndex = np.insert(selectedIndex,[-1],len(self.data_obj.tfs))

            moa_matrix = self.moa_matrix[TF_index,:]
            decoder_output = decoder(probe[selectedIndex,:])
        else:
            decoder_output = decoder(probe)
        mask = torch.logical_not(moa_matrix == 0)

        control_row = torch.masked_select(decoder_output[-1].repeat(moa_matrix.shape[0],1),mask)
        probe_rows = torch.masked_select(decoder_output[:-1],mask)

        diff = (probe_rows-control_row)
        #print("diff",diff)

        moa_vals = torch.masked_select(moa_matrix,mask)

        violated = torch.logical_not(torch.eq(torch.sign(diff),moa_vals))
        violated_count = torch.count_nonzero(violated.int().detach()).detach()

        loss = torch.tensor(0.0, requires_grad = True)
        if torch.any(violated):
            violated_values = (torch.abs(diff))*violated.int()
            lossL1 = self.beta * torch.sum(torch.abs(violated_values))
            lossL2 = (1-self.beta) * torch.sum(torch.square(violated_values))
            loss = self.moa_factor * (lossL1 + lossL2)

        return loss,violated_count


    def get_tf_gene_matrix(self):
        gene_index_dict = {}
        for i in range(len(self.data_obj.overlap_list)):
            gene_index_dict[self.data_obj.overlap_list[i]] = i

        x_coords = []
        y_coords = []
        moa_val = []
        for i in range(len(self.data_obj.tfs)):
            tf = self.data_obj.tfs[i]
            tf_info = self.data_obj.tf_gene_dict[tf]
            for gene in tf_info.keys():
                gene_index = gene_index_dict[gene]
                moa = tf_info[gene]
                if moa != 0:
                    x_coords.append(i)
                    y_coords.append(gene_index)
                    moa_val.append(moa)
        ind = [x_coords,y_coords]
        moa_matrix = torch.sparse_coo_tensor(ind,moa_val,([len(self.data_obj.tfs),len(self.data_obj.overlap_list)]))

        moa_matrix = moa_matrix.to_dense().to(device)

        return moa_matrix
