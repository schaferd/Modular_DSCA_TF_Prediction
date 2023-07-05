import pandas as pd
import numpy as np

ko_df = pd.read_csv("../knocktf.txt",delimiter='\t',low_memory=False)
overlap = pd.read_csv('../overlapping_exp.txt', delimiter=', ',engine='python') 
overlap_datasets = set(overlap["knocktf_ko"])

sample_ids = set(list(ko_df['Sample_ID']))
print(sample_ids)

for sample in sample_ids:
    if sample not in overlap_datasets: 
        df = ko_df[ko_df["Sample_ID"] == sample].copy(deep=True)
        TF = np.unique(df['TF']).tolist()[0]
        sample = np.unique(df['Sample_ID']).tolist()[0]
        try:
            print(TF,sample)
            #check if pvalues are satisfactory
            #print(df['Mean Expr. of Control'])
            df['Mean Expr. of Control'] = pd.to_numeric(df['Mean Expr. of Control'])
            df['Mean Expr. of Treat'] = pd.to_numeric(df['Mean Expr. of Treat'])
            df['Control'] = (df['Mean Expr. of Control'] - df['Mean Expr. of Control'].mean())/df['Mean Expr. of Control'].std()
            df['Treat'] = (df['Mean Expr. of Treat'] - df['Mean Expr. of Treat'].mean())/df['Mean Expr. of Treat'].std()
            df.drop(columns=["TF","FC","Log2FC","Rank","P_value","up_down"],inplace=True)
            control_df = df.drop(columns=['Mean Expr. of Control','Mean Expr. of Treat', 'Treat'])
            treated_df = df.drop(columns=['Mean Expr. of Control', 'Mean Expr. of Treat','Control'])
            control_df = control_df.pivot(index="Sample_ID", columns="Gene")
            treated_df = treated_df.pivot(index="Sample_ID", columns="Gene")
            control_df.columns = control_df.columns.get_level_values(1)
            treated_df.columns = treated_df.columns.get_level_values(1)
            treated_df.to_csv(sample+"."+str(TF)+".treated.csv")
            control_df.to_csv(sample+"."+str(TF)+".control.csv")
        except:
            print(TF,sample,"did not work :(")
