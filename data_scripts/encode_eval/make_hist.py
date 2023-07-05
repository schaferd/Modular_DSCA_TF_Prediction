import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

control_f = 'filtered_processed_control_df.csv'
treat_f = 'filtered_processed_treat_df.csv'

control_arr = pd.read_csv(control_f,sep='\t',index_col=0).to_numpy().flatten()
treat_arr = pd.read_csv(treat_f,sep='\t',index_col=0).to_numpy().flatten()

arr = np.vstack([control_arr,treat_arr]).flatten()

plt.hist(arr,bins=30)
plt.savefig("encode_hist.png")


