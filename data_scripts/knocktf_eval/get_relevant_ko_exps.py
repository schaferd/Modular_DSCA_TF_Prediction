import pandas as pd
import re
import os

data_files = os.listdir('/nobackup/users/schaferd/ae_project_data/ko_data/ko_datafiles/')

tfs = set()
with open('tfs.txt','r') as f:
    for line in f:
        line = line.strip()
        line = re.sub("'",'',line)
        tfs.add(line)


compatible_f = []

for f in data_files:
    tf = f.split('.')[1]
    if tf in tfs and "DataSet_01" in f:
        compatible_f.append(f)

compatible_f.sort()
print(compatible_f)

with open('compatible_exps.txt','w') as f:
    for exp in compatible_f:
        f.write(exp+"\n")





