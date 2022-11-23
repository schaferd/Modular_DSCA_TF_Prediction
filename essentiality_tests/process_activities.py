import numpy as np
import os
import sys
import pandas as pd

celllines_annotation = pd.read_csv('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B3_cell_lines/cellines2tissues_mapping.rdata')
GTEx_labels = list(celllines_annotation['GTEx'].drop("none"))
tcga_labels = list(celllines_annotation['Study.Abbreviation'].drop(''))
tcga_labels = [lower(x) for x in tcga_labels]
GTEx_tcga_labels = set(GTEx_labels+tcga_labels)


def process_activities(activities_dir):

    activities_files = [f for f in os.listdir(activities_dir)  if os.path.isfile(join(activities_dir,f))]
    activities_list = {} 
    for f in activities_files:
        df = pd.read_csv(f)
        f_name = set(f.split('/')[-1].split('.'))
        tissue = GTEx_tcga_labels.intersection(f_name)
        if len(tissue) == 1:
            which_samples_in_tissue = None
            if tissue in GTEx_labels:
                which_samples_in_tissue = celllines_annotation.index[celllines_annotation['GTEx'] == tissue]
            else:
                which_samples_in_tissue = celllines_annotation.index[celllines_annotation['Study.Abbreviation'] == upper(tissue)]
            cell_lines_in_tissue = celllines_annotation[which_samples_in_tissue,'COSMIC_ID']
            cell_lines_in_tissue = cell_lines_in_tissue[cell_lines_in_tissue['COMSIC_ID'] != 'NA']
            df = df[df['Sample'] == cell_lines_in_tissue['Sample']
        activities_list[f] = df
    return activities_list 

def apply_essentiality(essentiality_path):
    essentiality_df = pandas.read_csv('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B3_cell_lines/essentiality/DRIVE_ATARiS_data.csv')



if __name__ == '__main__':
    activities_dir = ''
    process_activities(activities_dir)


            


            

    

