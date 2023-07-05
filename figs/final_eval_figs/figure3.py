import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from data import final_const, pert_test, ko_test

sys.path.insert(1,'/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/figs/reg_test_figs/')
from shallow_l2_data_functions import create_l2_auc

sys.path.insert(1,'/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/figs/saved_eval_figs/')
from diff_rank_vs_auc import diff_plot 
from treat_rank_vs_auc import treat_plot 
from fc_rank_vs_auc import fc_plot

fig,ax = plt.subplots()
fig.set_figwidth(15)
fig.set_figheight(12)
label_font_size = 12
title_font_size = 15
subtitle_font_size=13

#fig.subplots_adjust(left=0.2,bottom=0.2,right=0.8,top=0.8,wspace=0.2,hspace=2)
fig.subplots_adjust(left=0.1,bottom=0.2,right=0.9,top=0.8,wspace=0.35,hspace=0)

subfigs1 = fig.subfigures(2,1)
subfigs1[0].subplots_adjust(left=0.1,bottom=0.2,right=0.9,top=0.8,wspace=0.35,hspace=0)

ax = subfigs1[0].subplots(1,3)

create_l2_auc(ax[0])
#ko_test(ax[1])
pert_test(ax[1])
final_const(ax[2])

ax = subfigs1[1].subplots(1,2)
#final_auc(subfigs1[1])
treat_plot(ax[0])
#diff_plot(ax[1])
fc_plot(ax[1])


fig.savefig('figure3.png',bbox_inches='tight',dpi=300)
