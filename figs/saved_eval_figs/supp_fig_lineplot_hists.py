import matplotlib.pyplot as plt
import seaborn as sns
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ko_tf_rank_scatter import knocktf_s_s_viper_scatter, knocktf_fc_g_viper_scatter,knocktf_fc_g_s_s_scatter,knocktf_s_s_scenic_scatter,knocktf_fc_g_scenic_scatter,dorothea_fc_g_s_s_scatter,dorothea_s_s_scenic_scatter,dorothea_fc_g_scenic_scatter,dorothea_fc_g_viper_scatter,dorothea_s_s_viper_scatter
sys.path.insert(1,'attribution_scores/')
from  attr_score_sum_vs_auc import * #dorothea_comp_other_methods_plot,dorothea_s_s_viper_consensus,dorothea_fc_g_viper_consensus, dorothea_fc_g_s_s_consensus, dorothea_s_s_scenic_consensus, dorothea_fc_g_scenic_consensus, knocktf_s_s_viper_consensus,knocktf_fc_g_viper_consensus,knocktf_comp_other_methods_plot, knocktf_fc_g_s_s_consensus, knocktf_s_s_scenic_consensus, knocktf_fc_g_scenic_consensus

fig,ax = plt.subplots(6,6,figsize=(30,40))
fig.subplots_adjust(left=0.1,bottom=0.2,right=0.9,top=0.8,wspace=0.3,hspace=0.5)

YMIN=0.48
YMAX=1
dorothea_lineplot_ax = ax[0]
[i.set_box_aspect(1.2) for i in dorothea_lineplot_ax]
dorothea_comp_other_methods_plot(dorothea_lineplot_ax[0],ymin=YMIN,ymax=YMAX,legend=True)
dorothea_fc_g_s_s_consensus(dorothea_lineplot_ax[1],ymin=YMIN,ymax=YMAX,legend=True)
dorothea_s_s_viper_consensus(dorothea_lineplot_ax[2],ymin=YMIN,ymax=YMAX,legend=True)
dorothea_fc_g_viper_consensus(dorothea_lineplot_ax[3],ymin=YMIN,ymax=YMAX,legend=True)
dorothea_s_s_scenic_consensus(dorothea_lineplot_ax[4],ymin=YMIN,ymax=YMAX,legend=True)
dorothea_fc_g_scenic_consensus(dorothea_lineplot_ax[5],ymin=YMIN,ymax=YMAX,legend=True)

dorothea_hists_ax = ax[1]
dorothea_comp_other_methods_hists(dorothea_hists_ax[0])
dorothea_s_s_fc_g_hists(dorothea_hists_ax[1])
dorothea_s_s_viper_hists(dorothea_hists_ax[2])
dorothea_fc_g_viper_hists(dorothea_hists_ax[3])
dorothea_s_s_scenic_hists(dorothea_hists_ax[4])
dorothea_fc_g_scenic_hists(dorothea_hists_ax[5])

dorothea_scatter_ax = ax[2]
[i.set_box_aspect(1) for i in dorothea_scatter_ax]
dorothea_fc_g_s_s_scatter(dorothea_scatter_ax[1])
dorothea_s_s_viper_scatter(dorothea_scatter_ax[2])
dorothea_fc_g_viper_scatter(dorothea_scatter_ax[3])
dorothea_s_s_scenic_scatter(dorothea_scatter_ax[4])
dorothea_fc_g_scenic_scatter(dorothea_scatter_ax[5])

divider0 = make_axes_locatable(dorothea_scatter_ax[0])
cax0 = divider0.append_axes("right",size="5%",pad=0.1)
dorothea_fc_g_viper_scatter(dorothea_scatter_ax[0],legend=True,cax=cax0)

YMAX=0.82
knocktf_lineplot_ax = ax[3]
[i.set_box_aspect(1.2) for i in knocktf_lineplot_ax]
knocktf_comp_other_methods_plot(knocktf_lineplot_ax[0],ymin=YMIN,ymax=YMAX,legend=True)
knocktf_fc_g_s_s_consensus(knocktf_lineplot_ax[1],ymin=YMIN,ymax=YMAX,legend=True)
knocktf_s_s_viper_consensus(knocktf_lineplot_ax[2],ymin=YMIN,ymax=YMAX,legend=True)
knocktf_fc_g_viper_consensus(knocktf_lineplot_ax[3],ymin=YMIN,ymax=YMAX,legend=True)
knocktf_s_s_scenic_consensus(knocktf_lineplot_ax[4],ymin=YMIN,ymax=YMAX,legend=True)
knocktf_fc_g_scenic_consensus(knocktf_lineplot_ax[5],ymin=YMIN,ymax=YMAX,legend=True)

knocktf_hists_ax = ax[4]
knocktf_comp_other_methods_hists(knocktf_hists_ax[0])
knocktf_s_s_fc_g_hists(knocktf_hists_ax[1])
knocktf_s_s_viper_hists(knocktf_hists_ax[2])
knocktf_fc_g_viper_hists(knocktf_hists_ax[3])
knocktf_s_s_scenic_hists(knocktf_hists_ax[4])
knocktf_fc_g_scenic_hists(knocktf_hists_ax[5])

knocktf_scatter_ax = ax[5]
[i.set_box_aspect(1) for i in knocktf_scatter_ax]
knocktf_fc_g_s_s_scatter(knocktf_scatter_ax[1])
knocktf_s_s_viper_scatter(knocktf_scatter_ax[2])
knocktf_fc_g_viper_scatter(knocktf_scatter_ax[3])
knocktf_s_s_scenic_scatter(knocktf_scatter_ax[4])
knocktf_fc_g_scenic_scatter(knocktf_scatter_ax[5])

divider1 = make_axes_locatable(knocktf_scatter_ax[0])
cax1 = divider1.append_axes("right",size="5%",pad=0.1)
knocktf_fc_g_viper_scatter(knocktf_scatter_ax[0],legend=True,cax=cax1)


fig.savefig("supp_fig_test.png",dpi=300,bbox_inches='tight')


