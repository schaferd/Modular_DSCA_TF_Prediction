import pandas as pd
import matplotlib as mpl
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
import re
import os, fnmatch
import sys
import matplotlib.pyplot as plt
import seaborn as sns

path = "/nobackup/users/schaferd/ae_project_outputs/model_eval/"

model_dirs = fnmatch.filter(os.listdir(path), "___*") 
print(model_dirs)

row_labels = ["S","T","FC"]
col_labels = ["S","G","FC"]
regex = '(?<=\_\_\_).+?(?=\_)'


fig,ax = plt.subplots(3,3, sharey=True, sharex = True)
#plt.tight_layout()
fig.supxlabel('Input')
fig.supylabel('Output')
plt.subplots_adjust(wspace=0.05, hspace=0.05)
fig.set_figwidth(8)
fig.set_figheight(8)

change_title = {'fc-fc':'FC-FC','fc-genefc':'FC-G','fc-shallow':'FC-S','tffc-fc':'T-FC','tffc-genefc':'T-G','tffc-shallow':'T-S', 'shallow-fc':'S-FC','shallow-genefc':'S-G','shallow-shallow':'S-S'}
model_dir_full_dict = {}
model_dir_enc_dict = {}
model_dir_dec_dict = {}

model_data =pd.DataFrame(columns=['encoder','decoder','input','output'])

for j,i in enumerate(model_dirs):
    model_type = change_title[re.findall(regex,i)[0]]
    print(i,model_type)
    model_dir_full_dict[i] = model_type
    model_dir_enc_dict[i] = model_type.split('-')[0]
    model_dir_dec_dict[i] = model_type.split('-')[1]
    run_dirs = fnmatch.filter(os.listdir(path+"/"+i), "fold*_cycle*") 
    print(run_dirs)
    input_gene_exp = []
    output_gene_exp = []
    for run in run_dirs:
        curr_dir = path+'/'+i+'/'+run+'/'
        input_gene_exp.extend(pd.read_pickle(curr_dir+"/"+fnmatch.filter(os.listdir(curr_dir),"test_input_cycle*_fold*_corr*.pkl")[0]))
        output_gene_exp.extend(pd.read_pickle(curr_dir+"/"+fnmatch.filter(os.listdir(curr_dir),"test_output_cycle*_fold*_corr*.pkl")[0]))
        break

    df = pd.DataFrame({'input':input_gene_exp,'output':output_gene_exp})
    df['encoder'] = model_dir_enc_dict[i]
    df['decoder'] = model_dir_dec_dict[i]
    print(df)
    print(i)
    model_data = pd.concat([model_data, df],axis=0)
    print(model_data)

ax_min = min(min(model_data["input"]),min(model_data["output"]))
ax_max = max(max(model_data["input"]),max(model_data["output"]))

#FC ENCODER ROW
fc_df = model_data[model_data["encoder"] == "FC"]
for i,a in enumerate(ax[0]):
    df = fc_df[fc_df["decoder"] == col_labels[i]]
    if i == 0:
        a.set_ylabel('FC Encoder')
    a.get_xaxis().set_visible(False)
    a.set_xlim(ax_min,ax_max)
    a.set_ylim(ax_min,ax_max)
    a.hist2d(x=df["input"], y=df["output"],bins=200,norm=mpl.colors.LogNorm())


#TF ENCODER ROW
tf_df = model_data[model_data["encoder"] == "T"]
for i,a in enumerate(ax[1]):
    df = tf_df[tf_df["decoder"] == col_labels[i]]
    print(str(1),str(i), "T",col_labels[i])
    print(df)
    a.get_xaxis().set_visible(False)
    if i == 0:
        a.set_ylabel('T Encoder')
    a.set_xlim(ax_min,ax_max)
    a.set_ylim(ax_min,ax_max)
    a.hist2d(x=df["input"], y=df["output"],bins=200,norm=mpl.colors.LogNorm())

#Shallow ENCODER ROW
s_df = model_data[model_data["encoder"] == "S"]
for i,a in enumerate(ax[2]):
    a.set_xlim(ax_min,ax_max)
    a.set_ylim(ax_min,ax_max)
    if i == 0:
        a.set_ylabel('S Encoder')
    a.set_xlabel(col_labels[i]+' Decoder')
    df = s_df[s_df["decoder"] == col_labels[i]]
    a.hist2d(x=df["input"], y=df["output"],bins=200,norm=mpl.colors.LogNorm())

fig.savefig('reconstruction_corr_hist.png')
