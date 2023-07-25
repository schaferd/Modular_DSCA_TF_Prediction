library(tidyverse)
library(decoupleR)



### Create an output folder-------------------------------------------------
outFolder <- paste0('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/viper_data/KOfilteredPKN_output_',file.path(format(Sys.time(), "%F %H-%M")))
#outFolder <- paste0('/nobackup/users/schaferd/ae_project_data/encode_ko_data/KOfilteredPKN_output_',file.path(format(Sys.time(), "%F %H-%M")))
#outFolder <- '/nobackup/users/schaferd/ae_project_data/ko_data/KOfilteredPKN_output_TFActivities/'
dir.create(outFolder)


### Read prior knowledge net------------------------------------------------
pkn <- read.delim('/nobackup/users/schaferd/ae_project_data/dorothea_tf_gene_relationship_knowledge/dorotheaSelectionAB_1.tsv') %>% select(-X)


### Load list of files and get unique KOed TFs from the names----------------
files <- list.files('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/viper_data/samples/')
#files <- list.files('/nobackup/users/schaferd/ae_project_data/encode_ko_data/ko_datafiles/')
tfs_koed <- NULL
for (i in 1:length(files)){
  file <- files[i]
  tfs_koed[i] <- str_split_fixed(file,"\\.",4)[1,2]
}
tfs_koed <- unique(tfs_koed)


### Filter PKN---------------------------------------------------------------
pkn_filtered <- pkn %>% filter(tf %in% tfs_koed) 
write.table(pkn_filtered, file = paste0(outFolder,'/','filtered_pkn.tsv'), quote=FALSE, sep = "\t", row.names = TRUE, col.names = NA)
pkn_filtered <- pkn_filtered %>% select(-confidence)

pkn_tfs <- unique(pkn_filtered$tf)
print(pkn_tfs)


### Run dorothea/viper for every file-----------------------------------------
minNrOfGenes = 10 
settings = list(verbose = F, minsize = minNrOfGenes)
for (file in files){
  gex <- data.table::fread(paste0('/nobackup/users/schaferd/ae_project_data/ko_data/filtered_data/relevant_data/viper_data/samples/',file),header = T) %>% column_to_rownames('Sample_ID')
  #gex <- data.table::fread(paste0('/nobackup/users/schaferd/ae_project_data/encode_ko_data/ko_datafiles/',file),header = T) %>% column_to_rownames('Sample_ID')
  gex <- rbind(gex,gex)
  print(gex)
  stop("hello")

  #VIPER_TF_activities = run_viper(t(gex), pkn_filtered,.source='tf',.target='target',.mor='mor',minsize=minNrOfGenes)
  #decoupleR_TF_activities = decouple(t(gex),pkn_filtered,.source='tf',.target='target',minsize=minNrOfGenes)
  SCENIC_TF_activities = run_aucell(t(gex),pkn_filtered,.source='tf',.target='target',minsize=minNrOfGenes)

  #decoupleR_TF_activities <- decoupleR_TF_activities %>% filter(statistic=='consensus') %>% pivot_wider_profile(id_cols=source, names_from=condition,values_from=score) %>% as.data.frame()
  #decoupleR_TF_activities <- decoupleR_TF_activities %>% t() %>% as.data.frame()

  #VIPER_TF_activities <- VIPER_TF_activities %>% select(source,condition,score) %>% spread('source','score') %>% column_to_rownames('condition')
  SCENIC_TF_activities <- SCENIC_TF_activities %>% select(source,condition,score) %>% spread('source','score') %>% column_to_rownames('condition')

  #write_csv(decoupleR_TF_activities[1,],paste0(outFolder,'/decoupleR_TFactivities_',file),append=FALSE,col_names=TRUE)
  #write_csv(VIPER_TF_activities[1,],paste0(outFolder,'/VIPER_TFactivities_',file),append=FALSE,col_names=TRUE)
  write_csv(SCENIC_TF_activities[1,],paste0(outFolder,'/SCENIC_TFactivities_',file),append=FALSE,col_names=TRUE)
}
