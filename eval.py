
    def get_correlation(self,data_loader):
        """
        run train data through final model then find correlation between input and output of final model
        """
        input_list = []
        output_list = []
        get_outputs_time = time.time()
        for samples, labels in data_loader:
            samples = samples.to(device)
            labels = labels.to(device)
            outputs = self.model(samples.float())
            print(outputs)
            print(outputs.shape)
            input_list.append(labels.tolist())
            output_list.append(outputs.tolist())
            if len(input_list) > 5:
                break
        print("get outputs time "+str(time.time()-get_outputs_time))
        avg_corr,corr_list = self.correlation(input_list,output_list)
        return avg_corr,corr_list,input_list,output_list

    def flatten_list(self,l):
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

    def correlation(self,ae_input,ae_output):
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

    def get_correlation_between_runs(self,data_loader):
            outputs = [] 
            for model in self.trained_models:
                for samples, labels in data_loader:
                    output = self.model(samples.float())
                    outputs.append(output.tolist())
            avg_pairwise_corr_list = []
            for i, output in enumerate(outputs):
                    if i != len(outputs)-1:
                            remaining_outputs = outputs[i+1:]
                            for output2 in remaining_outputs:
                                pairwise_corr, corr_list = self.correlation([output],[output2]) 
                                avg_pairwise_corr_list.append(pairwise_corr)
            avg_corr = sum(avg_pairwise_corr_list)/len(avg_pairwise_corr_list)
            print("corr between runs: "+str(avg_corr))

            if self.save_figs:
                    plt.clf()
                    ax = sns.swarmplot(data=corr_list,color=".4",alpha=0.1)
                    sns.boxplot(data=corr_list,ax=ax).set(title='corr btw runs')
                    plt.savefig(self.get_save_path()+'/corr_btw_runs_boxplot_'+str(self.model_type)+"_"+str(self.epochs)+'.png')
                    plt.clf()

            return avg_corr, avg_pairwise_corr_list

    def get_roc_curve(self,encoder,fold=0):
        tf_gene_dict = {tf:self.data_obj.tf_gene_dict[tf][0] for tf in self.data_obj.tf_gene_dict.keys()}
        ae_args = {
            'embedding':encoder,
            'overlap_genes': self.data_obj.overlap_list,
            'knowledge':tf_gene_dict,
            'data_dir':self.roc_data_path,
            'ae_input_genes':self.data_obj.input_genes,
            'tf_list':self.data_obj.tfs,
            'out_dir':self.get_save_path(),
            'fold':fold
        }
        obj = getROCCurve(ae_args=ae_args)
        #return obj.auc
        return obj.auc
