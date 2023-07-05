import sys
import subprocess
import os

download_path = "/nobackup/users/schaferd/ae_project_data/encode_ko_data/download_files/"
raw_counts = "/nobackup/users/schaferd/ae_project_data/encode_ko_data/raw_counts/"


files = set(os.listdir(download_path))
control_treated_pairs = {}
curr_dir = os.getcwd()

for f in files:
    if "files.txt" in f:
        beginning = '.'.join(f.split('.')[:-3])
        if beginning not in control_treated_pairs:
            control_treated_pairs[beginning] = [download_path+f]
        else:
            control_treated_pairs[beginning].append(download_path+f)

for pair in control_treated_pairs:
    files = set()
    for f in control_treated_pairs[pair]:
        with open(f,'r') as r:
            for line in r:
                if line.strip():
                    files.add(line)
    if len(files) < 4:
        print("REPEATS", pair, control_treated_pairs[pair])

for pair in control_treated_pairs:
    for f in control_treated_pairs[pair]:
        file_name = f.split('/')[-1]

        data_dir = raw_counts+'.'.join(file_name.split('.')[:-3])+'/'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        if 'treated' in file_name and 'files.txt' in file_name:
            treated_dir = data_dir+'treated/'
            if not os.path.exists(treated_dir):
                os.makedirs(treated_dir)
            os.chdir(treated_dir)
        elif 'control' in file_name and 'files.txt' in file_name:
            control_dir = data_dir+'control/'
            if not os.path.exists(control_dir):
                os.makedirs(control_dir)
            os.chdir(control_dir)

        subprocess.call("xargs -L 1 curl -O -J -L < "+str(f),shell=True)
        os.chdir(curr_dir)


