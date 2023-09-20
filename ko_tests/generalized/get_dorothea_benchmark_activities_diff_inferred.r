library(tidyverse)
library(data.table)
library(decoupleR)


contrast_folder = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/relevant_pos_neg_samples/set_kotf_to_0_inferred/'

### Create an output folder-------------------------------------------------
outFolder <- paste0('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/KOfilteredPKN_output_',file.path(format(Sys.time(), "%F %H-%M")))
dir.create(outFolder)


### Read prior knowledge net------------------------------------------------
pkn <- read.delim('/nobackup/users/schaferd/ae_project_data/dorothea_tf_gene_relationship_knowledge/dorotheaSelectionAB_1.tsv') %>% select(-X)

### Load list of files and get unique KOed TFs from the names----------------
files = list.files(path = contrast_folder, pattern = 'tive.csv', full.names = T)
tfs_koed <- NULL
for (i in 1:length(files)){
  file <- files[i]
  tfs_koed[i] <- tail(unlist(strsplit(files[i], split = '/')),1) %>% gsub('.csv', '', .) %>% strsplit(., split = '_') %>% unlist(.)
}
tfs_koed <- unique(tfs_koed)


### Filter PKN---------------------------------------------------------------
pkn_filtered <- pkn %>% filter(tf %in% tfs_koed) 
write.table(pkn_filtered, file = paste0(outFolder,'/','filtered_pkn.tsv'), quote=FALSE, sep = "\t", row.names = TRUE, col.names = NA)
pkn_filtered <- pkn_filtered %>% select(-confidence)


### Run dorothea/viper for every file-----------------------------------------
minNrOfGenes = 10 
settings = list(verbose = F, minsize = minNrOfGenes)
for (file in files){
  gex <- data.table::fread(file,header = T)
  gex <- gex %>% column_to_rownames('V1')

  VIPER_TF_activities = run_viper(t(gex), pkn_filtered,.source='tf',.target='target',.mor='mor',minsize=minNrOfGenes)
  SCENIC_TF_activities = run_aucell(t(gex),pkn_filtered,.source='tf',.target='target',minsize=minNrOfGenes)

  VIPER_TF_activities <- VIPER_TF_activities %>% select(source,condition,score) %>% spread('source','score') %>% column_to_rownames('condition')
  SCENIC_TF_activities <- SCENIC_TF_activities %>% select(source,condition,score) %>% spread('source','score') %>% column_to_rownames('condition')

  new_file_name = strsplit(file,split='/')[[1]]
  new_file_name = new_file_name[length(new_file_name)]

  write_csv(VIPER_TF_activities[1,],paste0(outFolder,'/VIPER_TFactivities_',new_file_name),append=FALSE,col_names=TRUE)
  write_csv(SCENIC_TF_activities[1,],paste0(outFolder,'/SCENIC_TFactivities_',new_file_name),append=FALSE,col_names=TRUE)
}
