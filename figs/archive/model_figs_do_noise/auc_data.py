import numpy as np
from scipy.stats import ttest_1samp, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#KO ROC AUC DATA

tffc_fc = [0.6326327571197612, 0.6334298085945891, 0.5250251551706595, 0.6231755435212716, 0.6046624684853761, 0.5796938417881087, 0.5986308803744446, 0.5809657324394297, 0.5752902737108682, 0.6390261274604018]
tffc_shallow = [0.6400888627601723, 0.6405919661733614, 0.6347073520932495, 0.6429718149031668, 0.6654531887711841, 0.6468157510938259, 0.6455382075951657, 0.6314739234152243, 0.6591785282246667, 0.6516771998055421] 
tffc_genefc =[0.6220449740534307, 0.685939107528462, 0.6943731557585555, 0.667301669851104, 0.6815807622299354, 0.6751930447366338, 0.6409537484030705, 0.6716826265389879, 0.6617392680693265, 0.6301624628325287]

shallow_fc = [0.5832268713751116, 0.5607568032017727, 0.6137296356174605, 0.5307119195938994, 0.5770709206227176, 0.5652395111417621, 0.5694169653254344, 0.6296367480299827, 0.5623791703881245, 0.5905642672213994]
shallow_shallow = [0.6504674904749521, 0.5614973262032086, 0.6375337757628516, 0.6187776282913703, 0.5822093588540548, 0.6199534205379249, 0.6432261930334309, 0.5995466416433958, 0.6378672937558649, 0.6686809646018699] 
shallow_genefc =[0.6343229584741834, 0.5836564877728911, 0.6689918712055262, 0.6574487569388701, 0.6658884580163028, 0.5824467784423013, 0.6083707363398942, 0.6436840736679065, 0.6414229347322247, 0.6416320900837753]


fc_fc= [0.5267718849984737, 0.5065064272874247, 0.4927247854744435, 0.4960486597098958, 0.5077387480073713, 0.483131903539813, 0.5054323862929757, 0.56504166148489, 0.5979921086251145, 0.48225571220223623]
fc_shallow = [0.6514341273699562, 0.6162903754621203, 0.6511571378503351, 0.6139444438163504, 0.6784434319566766, 0.6354309165526676, 0.6357079060722884, 0.6263637494205831, 0.6316548145300789, 0.6146962725124645]
fc_genefc = [0.6422991260698013, 0.6417507998778984, 0.5869181806876123, 0.658189279940306, 0.6878893398604877, 0.6183593175882692, 0.6344755853523419, 0.60463420424868, 0.605928706289358, 0.6676804106228307]


auc_dict = {'tf_fc':tffc_fc,'tf_shallow':tffc_shallow,'tf_gene':tffc_genefc, 'shallow_fc':shallow_fc, 'shallow_shallow':shallow_shallow, 'shallow_gene':shallow_genefc,'fc_fc':fc_fc,'fc_shallow':fc_shallow,'fc_gene':fc_genefc}

decoder = ['d_fc','d_shallow','d_gene']

fc_df = pd.concat([pd.DataFrame({'d_fc':fc_fc}),pd.DataFrame({'d_shallow':fc_shallow}),pd.DataFrame({'d_gene':fc_genefc})],axis=1)

fc_df = fc_df.melt(value_vars=decoder,value_name='AUC',var_name='decoder').dropna()
fc_df['encoder'] = 'e_fc'

shallow_df = pd.concat([pd.DataFrame({'d_fc':shallow_fc}),pd.DataFrame({'d_shallow':shallow_shallow}),pd.DataFrame({'d_gene':shallow_genefc})],axis=1)
shallow_df = shallow_df.melt(value_vars=decoder,value_name='AUC',var_name='decoder').dropna()
shallow_df['encoder'] = 'e_shallow'

tffc_df = pd.concat([pd.DataFrame({'d_fc':tffc_fc}),pd.DataFrame({'d_shallow':tffc_shallow}),pd.DataFrame({'d_gene':tffc_genefc})],axis=1)
tffc_df = tffc_df.melt(value_vars=decoder,value_name='AUC',var_name='decoder').dropna()
tffc_df['encoder'] = 'e_tf'


enc_sorter = ['e_shallow','e_tf','e_fc']
dec_sorter = ['d_shallow','d_gene','d_fc']

df = pd.concat([shallow_df,tffc_df,fc_df],axis=0)

df.decoder = df.decoder.astype("category")
df.decoder = df.decoder.cat.set_categories(dec_sorter)
df.encoder = df.encoder.astype("category")
df.encoder = df.encoder.cat.set_categories(enc_sorter)

print(df.decoder)
print(df.encoder)

