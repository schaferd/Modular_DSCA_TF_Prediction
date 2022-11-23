import pandas as pd




def read_hdf_file(hdf_file,save_path):
        hdf_keys = get_hdf_keys(hdf_file)
        data_df = pd.read_hdf(hdf_file,key=hdf_keys[1])
        for k in range(1,len(hdf_keys)):
                new_data = pd.read_hdf(hdf_file,key=hdf_keys[k])
                try:
                    data_df = data_df.join(new_data)
                except:
                        for gene in new_data.columns:
                                if gene in data_df.columns:
                                        new_data = new_data.drop(gene,axis=1)
                        data_df = data_df.join(new_data)
        print(data_df[:3])
        data_df.to_pickle(save_path+'agg_data.pkl')

def get_hdf_keys(hdf_file):
        with pd.HDFStore(hdf_file) as store:
                hdf_keys = store.keys()
                return hdf_keys


if __name__ == "__main__":
    read_hdf_file("/nobackup/users/schaferd/ae_project_data/hdf_gene_expression_data/unsupervised_O_z-score_train.h5","/nobackup/users/schaferd/ae_project_data/hdf_gene_expression_data/")



