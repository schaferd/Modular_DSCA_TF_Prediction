library(tidyverse)
library(dorothea)



### Create an output folder-------------------------------------------------
outFolder <- paste0('/nobackup/users/schaferd/ae_project_data/ko_data/KOfilteredPKN_output_',file.path(format(Sys.time(), "%F %H-%M")))
#outFolder <- '/nobackup/users/schaferd/ae_project_data/ko_data/KOfilteredPKN_output_TFActivities/'
dir.create(outFolder)


### Read prior knowledge net------------------------------------------------
pkn <- read.delim('/nobackup/users/schaferd/ae_project_data/dorothea_tf_gene_relationship_knowledge/dorotheaSelectionAB_1.tsv') %>% select(-X)


### Load list of files and get unique KOed TFs from the names----------------
files <- list.files('/nobackup/users/schaferd/ae_project_data/ko_data/ko_datafiles/')
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


### Run dorothea/viper for every file-----------------------------------------
minNrOfGenes = 10 
settings = list(verbose = F, minsize = minNrOfGenes)
for (file in files){
  gex <- data.table::fread(paste0('/nobackup/users/schaferd/ae_project_data/ko_data/ko_datafiles/',file),header = T) %>% column_to_rownames('Sample_ID')
  gex <- rbind(gex,gex)

  TF_activities = run_viper(t(gex), pkn_filtered, options =  settings)
  TF_activities <- as.data.frame(TF_activities) %>% select(rownames(gex)[1])
  TF_activities <- t(TF_activities)

  data.table::fwrite(as.data.frame(TF_activities),paste0(outFolder,'/TFactivities_',file),row.names = T)
}
