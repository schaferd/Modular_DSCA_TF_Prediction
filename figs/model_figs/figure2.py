import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fig2_2x3_split import make_model_boxplots
from moa_fig import create_moa_fig
from const_fig import create_const_figs

fig,ax = plt.subplots()
fig.set_figwidth(20)
fig.set_figheight(8)
label_font_size = 12
title_font_size = 15
subtitle_font_size=13

plt.subplots_adjust(left=0.2,bottom=0.2,right=0.8,top=0.8,wspace=0.2,hspace=2)

subfigs1 = fig.subfigures(1,2)

f = subfigs1[0]
make_model_boxplots(f,label_font_size,title_font_size,subtitle_font_size)

subfigs2 = subfigs1[1].subfigures(2,1)

#create_moa_fig(subfigs2[0],label_font_size,title_font_size,subtitle_font_size)
create_const_figs(subfigs2[1],label_font_size,title_font_size,subtitle_font_size)

fig.savefig('figure2.png',bbox_inches='tight',dpi=300)




