import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


m_base_path = '/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/microarray/'
r_base_path = '/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/eval_saved_models/outputs/rna_seq/'

m_shallow1 = m_base_path+'save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/'
m_shallow2 = m_base_path+'save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_14.26.53/'
m_deep1 = m_base_path+'save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.56/'
m_deep2 = m_base_path+'save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/'

r_shallow1 = r_base_path+'save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_12.58.48/'
r_shallow2 = r_base_path+'save_model_shallow-shallow_epochs100_batchsize128_enlr0.01_delr0.01_del20.01_enl20.01_moa1.0_rel_conn10_5-30_14.26.53/'
r_deep1 = r_base_path+'save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.56/'
r_deep2 = r_base_path+'save_model_fc-genefc_epochs100_batchsize128_enlr0.0001_delr0.01_del20.0001_enl20.0005_moa1.0_rel_conn10_5-30_12.59.31/'

r_deep_knocktf = pd.read_pickle(r_deep1+'knocktf_aucs.pkl')+pd.read_pickle(r_deep2+'knocktf_aucs.pkl')
r_shallow_knocktf = pd.read_pickle(r_shallow1+'knocktf_aucs.pkl')+pd.read_pickle(r_shallow2+'knocktf_aucs.pkl')

m_deep_knocktf = pd.read_pickle(m_deep1+'knocktf_aucs.pkl')+pd.read_pickle(m_deep2+'knocktf_aucs.pkl')
m_shallow_knocktf = pd.read_pickle(m_shallow1+'knocktf_aucs.pkl')+pd.read_pickle(m_shallow2+'knocktf_aucs.pkl')

rna_seq_group = [r_deep_knocktf,r_shallow_knocktf]
microarray_group = [m_deep_knocktf, m_shallow_knocktf]
x_labels = ['deep','shallow']

rna_seq_plot = plt.boxplot(rna_seq_group,positions=np.array(np.arange(len(rna_seq_group)))*2.0-0.35,widths=0.6)
microarray_plot = plt.boxplot(microarray_group,positions=np.array(np.arange(len(microarray_group)))*2.0+0.35,widths=0.6)

def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)

    plt.plot([], c=color_code, label=label)
    plt.legend()

define_box_properties(rna_seq_plot,'#D7191C', 'rna_seq')
define_box_properties(microarray_plot,'#2C7BB6', 'microarray')
plt.xticks(np.arange(0, len(x_labels) * 2, 2), x_labels)
plt.ylabel('ROC AUC')


plt.legend()
plt.savefig('microarray_vs_rnaseq.png',dpi=300)

