import numpy as np
import pandas as pd

control = [0.587048196176414, 0.6192298560785067, 0.6390204746130627]
do_2 = [0.6286983753716746, 0.6501565838712959, 0.5869407920769691]
do_4 = [0.622135419610858, 0.584617471820556, 0.5683881471096991]
do_6 = [0.6253236255101695, 0.5743292896632033, 0.5608415959118609] 
do_8 =  [0.6077206588958859, 0.6159568574691073, 0.6033849249867158]
do_1 = [0.48401940057206816, 0.4757832019988468, 0.5469299386100779]


##CORRS
control_corr = [0.5854478292648979, 0.5854066816761242, 0.5901549419875778]
do_2_corr = [0.5831292787666591, 0.5830790906410611, 0.6024441566149245]
do_4_corr = [0.5751362334485582, 0.5740113401253899, 0.581428734884754]
do_6_corr = [0.5760255547513518, 0.5831021568076169, 0.5852755764081371]
do_8_corr = [0.5788789267977319, 0.5781769633523999, 0.5767329395553357]
do_1_corr = [0.030245555813153188, 0.019658039543699192, 0.03280168572733664]

do_corrs = [control,do_2_corr,do_4_corr,do_6_corr,do_8_corr,do_1_corr]
do_aucs = [control,do_2,do_4,do_6,do_8,do_1]

x = [0,0.2,0.4,0.6,0.8,1]
do_corr_errors = []
do_auc_errors = []
do_corr_mean = []
do_auc_mean = []

for i in range(len(do_aucs)):
    do_corr_errors.append(np.std(np.array(do_corrs[i])))
    do_auc_errors.append(np.std(np.array(do_aucs[i])))
    do_corr_mean.append(np.mean(np.array(do_corrs[i])))
    do_auc_mean.append(np.mean(np.array(do_aucs[i])))

"""
fig, ax = plt.subplots()

print(do_auc_mean)
print(do_auc_errors)
ax.errorbar(x,do_auc_mean,yerr = do_auc_errors,ecolor='k',alpha=0.5, capsize=4)
ax.errorbar(x,do_corr_mean,yerr=do_corr_errors,ecolor='k',alpha=0.5,capsize=4)
fig.savefig('do_tests.png')

"""

