import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, ttest_ind
import seaborn as sns
import numpy as np

not_self_reg = [0.7429938372385486, -0.06852616113921219, 0.7651053329259699, 0.5649740037749115, -0.13809436949495119, 0.04447068230761537, -0.07744160477650504, 0.7857236406101348, 0.3074492449384866, 0.3120675631683669, 0.6180167099502961, 0.5639692984438607, 0.13690108947313756, 0.6190492083697083, -0.040525269863000736, 0.7240203643934785, 0.3388757332695643, 0.6929044324326012, 0.42116674141363697, 0.7972040642228393, 0.48897248628327494, 0.1307765948712585, 0.09257301211948665, 0.02653537681318829, 0.3639260697149907, 0.25677673123102807, 0.08093242237747894, -0.12369366406883818, 0.041672453182851524, 0.6169037055775987, 0.47065175992330727, 0.15270343479043028, -0.3983544551658408, 0.3523618064365109, -0.046666803004082984, 0.18062014479293106, 0.18280971609342633, 0.7087467011977446, 0.32060417408282943, -0.29963235841776104, 0.5425663969201379, 0.507349217042559, 0.7801283444520014, 0.7962037016746663, 0.68834211167875, 0.7799921331251479, 0.10517477662006136, 0.07280361533552118, 0.8538359551506727, 0.40165255173602527, 0.18733717496141578, 0.41996374739518655, 0.762872193729024, 0.5937888536695347, 0.7726277309570654, -0.026648512022152165, 0.12253750454893368, 0.40239424790608985, -0.10482668478257454, 0.11153001387443787, 0.26595388356392563, 0.38614093574225977, 0.37788065440769447, -0.01403095602225555, 0.7528053079275101, 0.24077405973353824, 0.8809514436230619, 0.3203235778266231, 0.17604656291303802, 0.2554275463932416, 0.6995464040201693, 0.19198423147267382, 0.0800452754258259, -0.41841368564539666, 0.7621461622548588, 0.5291424107140298, 0.43403068737115175, 0.8905191321275508, 0.78096589133968, 0.17375394811264708, 0.4749175253150207, -0.043874202912260775]

self_reg = [0.7631455738815587, 0.24668992724809385, 0.1345382039223046, 0.571282508275152, 0.5691353305980685, 0.8489594011069523, 0.08279728320537907, -0.07849676612682945, 0.7024215213176372, 0.3633029504085651, -0.041768052644548005, 0.4234456590592035, 0.7757020712561369, 0.7487091037522606, 0.5014089963758561, 0.5314718688766168, 0.29190768539432443, 0.2953227528687097, 0.1398261683098854, 0.7027924741076687, 0.34273004512738275, 0.8368804221758929, 0.4227017302091959, 0.3239240418914511, 0.09460609033126458, 0.17798616732346642]

normal =[0.04167245318285151, 0.3427300451273827, 0.6180167099502961, 0.42116674141363697, 0.2919076853943245, 0.8538359551506727, -0.04387420291226078, 0.19198423147267382, 0.25677673123102807, 0.13453820392230456, -0.07849676612682943, 0.501408996375856, 0.7799921331251479, 0.4749175253150205, 0.18733717496141578, 0.04447068230761536, 0.07280361533552118, 0.571282508275152, 0.24077405973353824, 0.11153001387443782, 0.48897248628327494, -0.13809436949495116, 0.5639692984438606, 0.08004527542582589, -0.06852616113921219, 0.692904432432601, 0.17604656291303794, 0.7972040642228393, 0.3639260697149907, 0.7487091037522606, 0.09257301211948665, 0.3206041740828295, -0.10482668478257454, 0.507349217042559, 0.7628721937290242, 0.7631455738815587, 0.4023942479060898, 0.09460609033126458, -0.026648512022152165, 0.7429938372385486, 0.8905191321275508, 0.7651053329259699, 0.26595388356392563, -0.041768052644548, 0.08279728320537907, 0.7024215213176374, 0.6169037055775987, 0.2554275463932416, 0.5691353305980684, 0.4227017302091959, 0.6995464040201694, -0.04052526986300072, 0.15270343479043025, 0.7757020712561369, 0.40165255173602527, 0.7801283444520014, 0.32032357782662296, -0.12369366406883818, 0.18280971609342633, -0.4184136856453966, 0.7240203643934785, 0.78096589133968, 0.5937888536695347, 0.10517477662006136, 0.3861409357422597, -0.014030956022255552, -0.07744160477650504, 0.17375394811264708, 0.6883421116787499, 0.1307765948712585, 0.7726277309570654, 0.1369010894731375, 0.7087467011977446, 0.836880422175893, 0.18062014479293106, 0.1225375045489337, 0.17798616732346645, 0.4234456590592035, 0.5291424107140299, 0.24668992724809394, 0.5649740037749115, -0.046666803004082984, 0.43403068737115175, 0.7962037016746663, 0.4706517599233073, 0.7857236406101348, 0.41996374739518655, 0.323924041891451, 0.5314718688766168, -0.29963235841776104, 0.08093242237747894, -0.39835445516584084, 0.37788065440769436, 0.02653537681318829, 0.7621461622548588, 0.3120675631683669, 0.2953227528687097, 0.3523618064365109, 0.3388757332695643, 0.3633029504085651, 0.5425663969201378, 0.8489594011069523, 0.8809514436230617, 0.7528053079275101, 0.3074492449384866, 0.1398261683098854, 0.7027924741076687, 0.6190492083697086]


print("not", len(not_self_reg))
print("reg",len(self_reg))
print("normal",len(normal))


stat, pval = ttest_ind(normal,self_reg)
print("all, self reg", pval)
stat, pval_r = ttest_ind(normal,not_self_reg)
print("all, not self reg", pval_r)
stat, pval_r = ttest_ind(self_reg,not_self_reg)
print("self reg, not self reg", pval_r)


fig, ax = plt.subplots()
ax.boxplot([normal,self_reg,not_self_reg])
#sns.swarmplot(data=[normal_corr,row_random_corr,col_random_corr,col_row_random])
labels = ["All TFs","Self-Regulated TFs","Non-Self-Regulated TFs"]
ax.set_title("RNA Seq TF Eval Correlation Distribution AE Self Regulation Test")
ax.set_ylabel("Correlation")
ax.set_xlabel("")
ax.set_xticks(np.arange(1, len(labels) + 1))
ax.set_xticklabels(labels)
fig.savefig("ae_self_reg.png")

