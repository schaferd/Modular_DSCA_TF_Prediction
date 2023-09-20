import matplotlib.pyplot as plt
import seaborn as sns
from extract_largest_signals import get_heatmap
from tf_hists import make_tf_hists

fig = plt.figure(figsize=(14,8))
fig.subplots_adjust(left=0.1,bottom=0.2,right=0.9,top=0.8,wspace=0.35,hspace=0.5)
subfigs = fig.subfigures(1,2)
#fig,ax = plt.subplots(1,2,figsize=(14,5))
make_tf_hists(subfigs[0].subplots())
get_heatmap(subfigs[1].subplots())
fig.savefig('figure4.png',bbox_inches='tight',dpi=300)
