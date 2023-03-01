import numpy as np
from scipy.stats import ttest_1samp, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#TEST CORRELATION BETWEEN INPUT AND OUTPUT GENE EXPRESSION

tffc_fc = [0.8772567890458024, 0.8778578649468892, 0.8704696069064908, 0.8748153832368852, 0.8724746391290308, 0.8751280585330752, 0.872501185121532, 0.8714592005659497, 0.8777215805475583, 0.8827922145081645]
tffc_shallow =[0.5346224719855646, 0.5388281152101673, 0.5241701271493658, 0.5364511820797377, 0.5316288880265186, 0.5431908459577361, 0.5098222097470765, 0.546086315761811, 0.5463151973679266, 0.5463656524635585]
tffc_genefc = [0.5816062984193962, 0.5725737671407398, 0.5816804821807674, 0.5866261208042002, 0.5963850427787163, 0.5816558478695313, 0.5829770391642942, 0.5621405167286067, 0.5714292600899594, 0.5788176424037352]

shallow_fc = [0.8744829496258809, 0.8673107046873498, 0.8732406768138973, 0.8812123503713071, 0.8800662940263941, 0.874754003145255, 0.8778148186804982, 0.8778911995487875, 0.8612848144526797, 0.8791583787634693]
shallow_shallow =[0.5003819567507531, 0.4982500995687465, 0.5090072559644453, 0.5127425284379168, 0.5118849797473842, 0.5102573003480126, 0.5010506034783822, 0.5206820098077093, 0.5137352056544702, 0.5200759159489197]
shallow_genefc =[0.56037912591617, 0.5760058667297312, 0.576855081372766, 0.5659939851399338, 0.5729794267411006, 0.572144466002336, 0.572369215568166, 0.5758952091616281, 0.5616042204368278, 0.5688833845585333]

fc_fc= [0.8993481920318575, 0.8952683640731661, 0.8954643533513292, 0.8936257820218905, 0.8962139691485084, 0.8990099143076941, 0.8932232910594017, 0.901175050530057, 0.8912360115771146, 0.8922554313462934]
fc_shallow = [0.5264139111291369, 0.5390885397260173, 0.5277932902108643, 0.5406711397508694, 0.5272908198229449, 0.5301862778145328, 0.5298466295003286, 0.5348114398607373, 0.5171236452881907, 0.5340345640290655]
fc_genefc =[0.5907796271126105, 0.5901685477811704, 0.580421001196783, 0.5742076481578186, 0.5807062202622861, 0.5829509735692452, 0.5958910106635921, 0.5899577410846276, 0.5834560205088593, 0.5920445651523675]

corr_dict = {'tf_fc':tffc_fc,'tf_shallow':tffc_shallow,'tf_gene':tffc_genefc, 'shallow_fc':shallow_fc, 'shallow_shallow':shallow_shallow, 'shallow_gene':shallow_genefc,'fc_fc':fc_fc,'fc_shallow':fc_shallow,'fc_gene':fc_genefc}

decoder = ['d_fc','d_shallow','d_gene']

fc_df = pd.concat([pd.DataFrame({'d_fc':fc_fc}),pd.DataFrame({'d_shallow':fc_shallow}),pd.DataFrame({'d_gene':fc_genefc})],axis=1)

fc_df = fc_df.melt(value_vars=decoder,value_name='Corr',var_name='decoder').dropna()
fc_df['encoder'] = 'e_fc'

shallow_df = pd.concat([pd.DataFrame({'d_fc':shallow_fc}),pd.DataFrame({'d_shallow':shallow_shallow}),pd.DataFrame({'d_gene':shallow_genefc})],axis=1)
shallow_df = shallow_df.melt(value_vars=decoder,value_name='Corr',var_name='decoder').dropna()
shallow_df['encoder'] = 'e_shallow'

tffc_df = pd.concat([pd.DataFrame({'d_fc':tffc_fc}),pd.DataFrame({'d_shallow':tffc_shallow}),pd.DataFrame({'d_gene':tffc_genefc})],axis=1)
tffc_df = tffc_df.melt(value_vars=decoder,value_name='Corr',var_name='decoder').dropna()
tffc_df['encoder'] = 'e_tf'




enc_sorter = ['e_shallow','e_tf','e_fc']
dec_sorter = ['d_shallow','d_gene','d_fc']


df = pd.concat([fc_df,shallow_df,tffc_df],axis=0)

df.decoder = df.decoder.astype("category")
df.decoder = df.decoder.cat.set_categories(dec_sorter)
df.encoder = df.encoder.astype("category")
df.encoder = df.encoder.cat.set_categories(enc_sorter)

