import matplotlib.pyplot as plt
import seaborn as sns
import sys
from set_kotf_0_boxplots import plot_ktf_vanilla, plot_dorothea_vanilla
from ko_tf_rank_scatter import *#knocktf_s_s_viper_scatter, knocktf_fc_g_viper_scatter,knocktf_fc_g_s_s_scatter,dorothea_fc_g_viper_scatter,dorothea_s_s_viper_scatter
sys.path.insert(1,'attribution_scores/')
#from  attr_score_sum_vs_auc import *#dorothea_comp_other_methods_plot,dorothea_s_s_viper_consensus,dorothea_fc_g_viper_consensus,knocktf_s_s_consensus,knocktf_fc_g_consensus,knocktf_comp_other_methods_plot
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
fig = plt.figure(figsize=(14,14))
subfigs = fig.subfigures(2,1,hspace=0.05,height_ratios=[3,1])
plt.subplots_adjust(top=0.95,bottom=0.2,wspace=0.2, hspace=0.4)
#subfigs[0].set_box_aspect(3/4)
#subfigs[1].set_box_aspect(1/4)
top_ax = subfigs[0].subplots(2,4,sharey=True)
bottom_ax = subfigs[1].subplots(1,4,sharey=True,sharex=True)



dorothea_row = top_ax[0]
#[i.set_box_aspect(3) for i in dorothea_row]
plot_dorothea_vanilla(dorothea_row[0])
dorothea_comp_other_methods_plot(dorothea_row[1])
dorothea_s_s_consensus(dorothea_row[2])
dorothea_fc_g_consensus(dorothea_row[3])

knocktf_row = top_ax[1]
#[i.set_box_aspect(3) for i in knocktf_row]
plot_ktf_vanilla(knocktf_row[0],legend=True)
knocktf_comp_other_methods_plot(knocktf_row[1],legend=True)
knocktf_s_s_consensus(knocktf_row[2],legend=True)
knocktf_fc_g_consensus(knocktf_row[3],legend=True)

divider = make_axes_locatable(bottom_ax[3])
cax = divider.append_axes("right", size="5%", pad=0.1)

subfigs[1].subplots_adjust(top=0.95,bottom=0.2,wspace=0.3, hspace=0.4)
#[i.set_box_aspect(1) for i in bottom_ax]
#bottom_ax[4].set_box_aspect(15)
knocktf_s_s_viper_scatter(bottom_ax[0])
knocktf_fc_g_viper_scatter(bottom_ax[1])
#knocktf_s_s_viper_con_scatter(bottom_ax[2])
dorothea_s_s_viper_scatter(bottom_ax[2])
dorothea_fc_g_viper_scatter(bottom_ax[3],legend=True,cax=cax)
#knocktf_fc_g_s_s_scatter(bottom_ax[3])
#knocktf_s_s_viper_viper_con_scatter(bottom_ax[3],legend=True,cax=cax)
"""

#fig = plt.figure(figsize=(13,5))
#fig.subplots_adjust(left=0.1,bottom=0.2,right=0.9,top=0.8,wspace=0.2,hspace=0.2)
#subfigs = fig.subfigures(1,2)
#subfigs[0].suptitle('Perturbation Validation',fontsize=16)
#subfigs[1].suptitle('KnockTF Validation',fontsize=16)
#top_ax = subfigs[0].subplots(1,1,sharey=False,sharex=False)
#bottom_ax = subfigs[1].subplots(1,1,sharey=False,sharex=False)
fig, ax = plt.subplots(1,2,sharey=True,figsize=(13,6))

plot_dorothea_vanilla(ax[0],legend=True,ymin=0.48,ymax=0.9)
#dorothea_comp_other_methods_plot(top_ax[1],legend=True,ymin=0.48,ymax=1)
#dor_recon_corr_hist(top_ax[2])
#ktf_recon_corr_hist(bottom_ax[2])
plot_ktf_vanilla(ax[1],legend=True,ymin=0.48,ymax=0.9)
#knocktf_comp_other_methods_plot(bottom_ax[1],legend=True,ymin=0.48,ymax=0.7)

fig.savefig('figure3.png',bbox_inches='tight',dpi=300)

"""
fig = plt.figure(figsize=(18,11))
fig.subplots_adjust(left=0.1,bottom=0.2,right=0.9,top=0.8,wspace=0.1,hspace=0.2)
subfigs = fig.subfigures(2,1)
subfigs[0].suptitle('Perturbation Validation',fontsize=16)
subfigs[1].suptitle('KnockTF Validation',fontsize=16)

top_ax = subfigs[0].subplots(1,3,sharey=True)
bottom_ax = subfigs[1].subplots(1,3,sharey=False,sharex=False)
#[ax.set_zorder(0) for ax in bottom_ax]
#[ax.set_zorder(1) for ax in top_ax]

[i.set_box_aspect(1.2) for i in top_ax]
bottom_ax[0].set_box_aspect(1)
bottom_ax[1].set_box_aspect(1.2)
bottom_ax[2].set_box_aspect(1)
divider2 = make_axes_locatable(bottom_ax[2])
divider0 = make_axes_locatable(bottom_ax[0])

cax0 = divider0.append_axes("right", size="5%", pad=0.1)
cax2 = divider2.append_axes("right", size="5%", pad=0.1)
#bottom_ax[2].set_box_aspect(10)

dorothea_fc_g_viper_scatter(bottom_ax[0],legend=True,cax=cax0)
print("knocktf")
knocktf_comp_other_methods_plot(bottom_ax[1],legend=True,ymin=0.48,ymax=0.7)
knocktf_fc_g_viper_scatter(bottom_ax[2],legend=True,cax=cax2)
print("dorothea")
plot_dorothea_vanilla(top_ax[0],legend=True)
dorothea_comp_other_methods_plot(top_ax[1],legend=True,ymin=0.48,ymax=1)
dorothea_s_s_viper_consensus(top_ax[2],legend=True,ymin=0.48,ymax=1)

fig.savefig('figure3.png',bbox_inches='tight',dpi=300)

plot_ktf_vanilla(bottom_ax[0])
"""
