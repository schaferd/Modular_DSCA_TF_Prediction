import matplotlib.pyplot as plt
import seaborn as sns
from extract_largest_signals import *#get_heatmap, get_comb_heatmap
from tf_hists import make_tf_hists

fig,ax = plt.subplots(1,3,figsize=(20,6))
fig.subplots_adjust(left=0.1,bottom=0.2,right=0.9,top=0.8,wspace=0.35,hspace=0.5)
#fig,ax = plt.subplots(1,2,figsize=(14,5))
get_model_num_filter_line_plot_noAB(ax[1])
make_tf_hists(ax[0])
get_heatmap(ax[2])
#get_comb_heatmap(ax[2])
#ax[2].set_box_aspect(0.3)
#get_comb_heatmap(ax[2])

fig.savefig('figure4.png',bbox_inches='tight',dpi=300)
