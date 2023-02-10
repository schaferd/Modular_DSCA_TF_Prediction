import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from auc_data import auc_dict
from corr_data import corr_dict

const_dict = {'fc_fc':0.111,'fc_gene':0.3434,'fc_shallow':0.5372,'shallow_fc':0.3369,'shallow_gene':0.5314,'shallow_shallow':0.4399,'tf_gene':0.4622, 'tf_fc':0.3435, 'tf_shallow':0.50366}

keys = const_dict.keys()
auc_std = [np.std(auc_dict[col]) for col in keys]
const = [const_dict[col] for col in keys]
df = pd.DataFrame({"model":keys,"auc_std":auc_std,"const":const})
corr = df["const"].corr(df["auc_std"])
a, b = np.polyfit(df["const"],df["auc_std"], 1)
x = np.arange(min(df["const"]),max(df["const"]),0.005)

plt.clf()
fig,ax = plt.subplots()
plt.plot(x, a*x+b, color='0.8',linestyle='dashed',zorder=1)
markers = ["." , "," , "o" , "v" , "^" , "<", ">"]
for i,col in enumerate(auc_dict.keys()):
    if col in const_dict:
        plt.scatter(const_dict[col],np.std(auc_dict[col]),marker=markers[i%len(markers)],label=col,edgecolors='black',zorder=2)

plt.legend(loc='best')
plt.xlabel('Consistency')
plt.ylabel('KO ROC AUC Std')
plt.title('Model Type Consistency vs. KO ROC AUC Std, corr: '+str(round(corr,2)))
plt.savefig('const_vs_aucStd.png')
