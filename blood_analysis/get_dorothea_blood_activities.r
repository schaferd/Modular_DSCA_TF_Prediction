#!/usr/bin/env Rscript
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("dorothea")
library(dorothea)
library(ggplot2)
library(dplyr)

net <- dorothea::dorothea_hs

#data_file = "/nobackup/users/schaferd/blood_analysis_data/SCP43/expression/blood_data.csv"
data_file = "/nobackup/users/schaferd/ae_project_data/hdf_gene_expression_data/train_tf_agg_data_gene_id.csv"
data <- data.table::fread(data_file)
samples <- data$V1
data <- data[,-1]
genes <- colnames(data)
data <- t(data)
rownames(data) <- genes
colnames(data) <- samples

confidenceFilter = is.element(net$confidence, c('A', 'B'))
net = net[confidenceFilter,]

activities = run_viper(data, net, options=list(method="scale",minsize=4,eset.filter=FALSE,cores=1,verbose=FALSE))
message("regulon activities")
#analysis_name = gsub('contrasts/', 'activities/global/viper_', DE_file)
analysis_name = gsub('csv', 'viper_pred.csv', data_file)
print(analysis_name)
write.csv(activities,analysis_name)



