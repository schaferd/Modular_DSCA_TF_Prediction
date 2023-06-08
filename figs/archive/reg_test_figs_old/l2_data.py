import numpy as np

control = [0.587048196176414, 0.6192298560785067, 0.6390204746130627]
l2_1 = [0.5189257328916576, 0.5094006851250975, 0.5457654520582017]
l2_2 = [0.5359012334512893, 0.46672734056144083, 0.48942352262834793]
l2_3 = [0.5, 0.5, 0.5]
l2_4 = [0.7029485251721292, 0.7002973397700422, 0.7024115046749048]
l2_5 = [0.6178222970910447, 0.6496139105267322, 0.6281161320957367, 0.6219884455800387, 0.6289301421125821, 0.6768888989383953]


control_corr = [0.5854478292648979, 0.5854066816761242, 0.5901549419875778]
l2_1_corr = [0.021647544019849193, 0.01192522967987772, 0.027319078916817043]
l2_2_corr = [0.011957903754765166, 0.028677781660075632, 0.020331736576200906]
l2_3_corr = [0.03311047227198914, 0.02449328961964371, 0.01653631527410735]
l2_4_corr = [0.31242538300167066, 0.3104028714884993, 0.3017666529066633]
l2_5_corr = [0.5544965222172251, 0.5517454656314579, 0.5459204285028084, 0.5481111540451362, 0.5461178157684472, 0.5434722583912515]



l2_corrs = [control,l2_5_corr,l2_4_corr,l2_3_corr,l2_2_corr,l2_1_corr]
l2_aucs = [control,l2_5, l2_4,l2_3,l2_2,l2_1]

l2_corr_errors = []
l2_auc_errors = []
l2_corr_mean = []
l2_auc_mean = []

for i in range(len(l2_aucs)):
    l2_corr_errors.append(np.std(np.array(l2_corrs[i])))
    l2_auc_errors.append(np.std(np.array(l2_aucs[i])))
    l2_corr_mean.append(np.mean(np.array(l2_corrs[i])))
    l2_auc_mean.append(np.mean(np.array(l2_aucs[i])))
