import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from data import final_auc, final_const

sys.path.insert(1,'/home/schaferd/ae_project/Modular_DSCA_TF_Prediction/figs/reg_test_figs/')
from shallow_l2_data_functions import create_l2_auc


fig,ax = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(12)
label_font_size = 12
title_font_size = 15
subtitle_font_size=13

#fig.subplots_adjust(left=0.2,bottom=0.2,right=0.8,top=0.8,wspace=0.2,hspace=2)
fig.subplots_adjust(left=0.1,bottom=0.2,right=0.9,top=0.8,wspace=0.35,hspace=0)

subfigs1 = fig.subfigures(2,1)
subfigs1[0].subplots_adjust(left=0.1,bottom=0.2,right=0.9,top=0.8,wspace=0.35,hspace=0)

ax = subfigs1[0].subplots(1,2)

create_l2_auc(ax[0])
final_const(ax[1])

final_auc(subfigs1[1])

fig.savefig('figure3.png',bbox_inches='tight',dpi=300)
