import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, ttest_ind
import pandas as pd
import numpy as np
import seaborn as sns

dorothea_corr = [0.35246566037119187, -0.021349486576576048, 0.019960860979961866, 0.23676812516756318, 0.22309531983510705, -0.04742903766949839, 0.2227433687848726, 0.29764545615833266, -0.007060457653250366, 0.1328077581370403, 0.27342328183987336, -0.033008590382018116, 0.20683611223483536, 0.18524822887961737, 0.021727589580354163, 0.10442890035272984, 0.30071709896931414, 0.102847267669314, 0.21276767219121973, -0.39750835663709216, 0.22988445054644302, 0.26831028087445025, 0.018361589746986835, 0.18795836871080585, 0.1808072015371219, 0.07590555881891096, -0.19882396452986845, 0.2694571607903621, -0.2586785714598658, 0.3010708353177063, 0.2396392875490737, -0.49628722570241923, 0.055389678891741644, 0.04797209450252254, 0.2509888882345453, 0.37587646989387674, 0.036007044056303075, 0.33241804192144164, 0.107488620911397, 0.509889341791496, 0.08940443822185712, -0.19826546262363004, 0.02328521157757688, 0.16541231073078688, 0.1822127505836216, 0.25486543421380836, 0.08180351899426345, 0.09847635864874849, 0.3054716702584317, 0.36550155763434333, -0.02233366977139505, 0.5231732945341424, -0.07607192882740775, 0.39290252677500603, 0.17710328335436412, 0.30902094505897076, -0.31290132153776057, 0.38995680502603236, 0.30530503235711964, 0.28822570299872985, 0.2267801121175574, 0.04933636042744077]

random_corr = [-0.02062568130738437, 0.003076696445974432, 0.004126578397474759, 0.02008302494339811, 0.0005905294970271485, 0.009286259323169496, -0.0010755475229120913, 0.0014871353982082276, -0.007755897685029311, -0.0027053660379791366, 0.0008797303673949923, -0.0006809164722466704, -0.003206835896977769, -0.001278284338429293, 0.011810250886070786, -0.00595027143248823, -0.01615643393786585, -0.007960536076834869, 0.012335366066215991, 0.006030048983835143, 0.015761766468255774, 0.004025501654062505, 0.007828722718784122, 0.0030293742945118737, -0.01832896338797692, -0.011182922512189983, 0.0005318132430227805, 0.014187565055960893, -0.00556464887810216, -0.007756463494873725, 0.0017875270286915117, 0.0058985400398409785, 0.007271735628269486, 0.006570292513261277, 0.005852380125824194, 0.011781836091351432, -0.011295839744032773, 0.00410534049367995, -0.004999754693862719, -0.010557911755492175, -0.0003469498396926585, -0.004335965486672369, -0.0014193130032927519, 0.0058028708713752895, 0.009549060432996673, 0.0002915052972809344, -0.0014681559513186138, 0.012187616061724305, 0.008828310546793793, -0.0075088935161885745, 0.0022956093279326876, 0.0009471095062981872, 0.007559467573100459, 0.0024618202585596372, -0.0064761538658077355, -0.013069872694217638, 0.009345687992256256, 0.003988707873266895, -0.009475653072600503, 0.0162776664494473, 0.004538500450302106, 0.004455542529269112, -0.010444185580011244, 0.0014762048165098383, 0.0001737329696634683, 0.009239015252273894, 0.0009717678669191453, -0.01815184722812113, 0.0009434187008641162, 0.00709737791306504, -0.001904666925755277, -0.0037813284879242943, 0.006299623583666873, 0.004162650762495932, 0.0016579723626857837, 0.007785601956226589, -0.001558695904931491, -0.004188088209590842, -0.0034119020278216582, 0.012756649319376628, -0.0028085787181364764, 0.0056925679718160685, 0.006019355473043555, 0.003205189451037485, -0.00782217636595973, 0.00398715349010017, -0.008318513691182083, 0.00923384683733694, -0.00019791146005826145, 0.0033140966348642526, -0.012269079611709624, 0.009481111219903953, -0.004225421192092142, 0.008659847218514287, -0.001236831051789259, -0.002569849151404465, 0.007729044411052797, -0.0035563968824029175, -0.01076701567380482, 0.0020088160152097697, -0.003549821741308985, 0.011978930840956505, -0.01465548572402661, 0.003382708709771842, -0.008753041316194797, -0.0060457774854447725, -0.0027139674300761204, 0.0024473578370555853]

ae_corr = [0.025952946713696787, 0.13789384470446645, 0.12571187248281843, 0.3314594781208889, 0.20802776523314986, 0.5825854429997708, 0.10099746771953905, 0.7593787965420865, 0.30386659859674964, -0.19698805789101878, 0.13398807589219952, 0.786844154533683, 0.15504392334833494, -0.017726612394678064, 0.48907450861999513, 0.697521500071042, -0.13049053757692275, 0.0025896511386339655, 0.07060163070899722, 0.6048095747680985, 0.21523981427916064, 0.7292840380081561, 0.022249500024452844, 0.7814770240331376, 0.008506969705866218, -0.0466155217838706, 0.505480112130044, 0.44139628050567503, 0.05081539320780084, -0.0041956815781074635, 0.14068580893584157, 0.3459956138066298, 0.7561199353187618, 0.408245924247501, 0.09680410384482512, 0.13106337036688498, 0.44129093938328445, 0.6447976281715555, 0.6779174241885239, 0.21028501100126523, 0.3729454576744853, 0.1270366810565182, 0.5556482132302462, 0.5932805310766736, 0.37754554778750654, 0.14026495444539233, -0.14179624282543296, 0.14950881572736663, 0.29001810888441215, 0.25439920316674114, 0.8700130705757699, 0.8866561580784205, 0.7967815894543704, 0.4253697022382942, 0.16079244484509733, 0.019163843048990933, 0.744906824600928, 0.5413595648568236, -0.2846283501611536, 0.8582598035154949, 0.44600993149646695, 0.8077068197420254, 0.06872293979949026, 0.2280523851597305, 0.3460074873556786, 0.7696804762252768, 0.3774114494000159, 0.46483294064651415, 0.48982330257156476, 0.03053532365101878, -0.08708359983434195, 0.5247370770698492, 0.4921262255856108, 0.13405519091879906, 0.13928874045252426, 0.7138136227002246, 0.09350254554810374, 0.8591261346485547, 0.5895848679556692, 0.2647508158699074, 0.5435366269805525, 0.20860540412294684, -0.16874758834715356, 0.35724939252382154, 0.7312562032349882, 0.2989431122994285, 0.10601956283690646, -0.30347356708174805, 0.7577995497720827, 0.8088384322884736, 0.35112237921145917, 0.18354189328060072, 0.6140692192339594, 0.706917309344836, 0.2416673401478896, 0.7114245774033734, 0.7186059387012103, 0.7997844534915439, 0.4780379832459704, -0.012176157331156689, -0.5397100435906506, 0.7681157369251808, 0.4812523896017476, 0.5051337654462211, 0.7826400765025301, -0.0596854008898991, 0.12484367837151526, 0.7688816805836661]

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


stat, pval_d = ttest_ind(ae_corr,dorothea_corr)
stat, pval_r = ttest_ind(ae_corr,random_corr)

print("pval d",pval_d)
print("pval r",pval_r)

fig, ax = plt.subplots()
ax.boxplot([ae_corr,dorothea_corr,random_corr])
labels = ["AE","DoRothEA","Random"]
ax.set_title("RNA Seq TF Eval Correlation Distribution")
ax.set_ylabel("Correlation")
ax.set_xlabel("Method")
ax.set_xticks(np.arange(1, len(labels) + 1))
ax.set_xticklabels(labels)

fig.savefig("corr_boxplot.png")