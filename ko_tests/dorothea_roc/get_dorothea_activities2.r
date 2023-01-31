#!/usr/bin/env Rscript
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("dorothea")
library(dorothea)
library(ggplot2)
library(dplyr)

#net <- dorothea::dorothea_hs

#ENV VARIABLES
# Set enviroment folders
contrast_folder = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/contrasts'
design_folder = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/design'
#regulons_folder ='/nobackup/users/schaferd/ko_eval_data/data/TF_target_sources'
regulons = "/nobackup/users/schaferd/ae_project_data/dorothea_tf_gene_relationship_knowledge/dorotheaSelectionAB_1.tsv"
outdir = '/nobackup/users/schaferd/ko_eval_data/results/regulons_QC/B1_perturbations/'
DE_folder = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/contrasts/'
perturbation_files = list.files(path = contrast_folder, pattern = 'rdata', full.names = T)

#net <- data.table::fread(regulons)
net <- dorothea::dorothea_hs
#net = get(load(paste(regulons_folder, '/', n, sep = '')))
print(net)

confidenceFilter = is.element(net$confidence, c('A', 'B'))
net = net[confidenceFilter,]


for (DE_file in perturbation_files){

  DEGs = get(load(DE_file))
  DEGs = subset(DEGs, Symbol != "" )
  DEGs = subset(DEGs, ! duplicated(Symbol))

  myStatistics = matrix(DEGs$logFC, dimnames = list(DEGs$Symbol, 'logFC') )
  myPvalue = matrix(DEGs$P.Value, dimnames = list(DEGs$Symbol, 'P.Value') )
  # Although Gene Expression Signature (GES) can be defined by the t-statistic,
  # to be consistent with the z-score based null model for msVIPER (see section 6.2 in its vignette),
  # we will estimate z-score values for the GES as indicated in the vignette.
  mySignature = (qnorm(myPvalue/2, lower.tail = FALSE) * sign(myStatistics))[, 1]
  mySignature = mySignature[order(mySignature, decreasing = T)]
  print(mySignature)

  perturbation_id = tail(unlist(strsplit(DE_file, split = '/')),1) %>% gsub('.rdata', '', .) %>% strsplit(., split = '\\.') %>% unlist(.)
  activities = run_viper(DEGs, net, options=list(method="scale",minsize=4,eset.filter=FALSE,cores=1,verbose=FALSE))
  message("regulon activities")
  message(activities$Regulon)
  analysis_name = gsub('contrasts/', 'activities/global/viper_', DE_file)
  analysis_name = gsub('rdata', 'viper_pred.csv', DE_file)
  print(analysis_name)
  write.csv(activities,analysis_name)
}





##########################################################################################
#FUNCTIONS###########################
##########################################################################################






###################################################################
#COMPUTE ACTIVITIES USING VIPER
###################################################################

DEs2activities = function(DE_file, design_folder, networks){
  #source('code/lib/contrast.r')

  DEGs = get(load(DE_file))
  DEGs = subset(DEGs, Symbol != "" )
  DEGs = subset(DEGs, ! duplicated(Symbol))

  perturbation_id = tail(unlist(strsplit(DE_file, split = '/')),1) %>% gsub('.rdata', '', .) %>% strsplit(., split = '\\.') %>% unlist(.)
  design_file = list.files(design_folder, recursive = T, full.names = T) %>% grep(perturbation_id[1], ., value = T) %>% grep(perturbation_id[2], ., value = T)
  tissue = read_desigfile(design_file)$GTEx_tissue

  myStatistics = matrix(DEGs$logFC, dimnames = list(DEGs$Symbol, 'logFC') )
  myPvalue = matrix(DEGs$P.Value, dimnames = list(DEGs$Symbol, 'P.Value') )
  # Although Gene Expression Signature (GES) can be defined by the t-statistic,
  # to be consistent with the z-score based null model for msVIPER (see section 6.2 in its vignette),
  # we will estimate z-score values for the GES as indicated in the vignette.
  mySignature = (qnorm(myPvalue/2, lower.tail = FALSE) * sign(myStatistics))[, 1]
  mySignature = mySignature[order(mySignature, decreasing = T)]

  message('- Running VIPER ...')

  #need to get my data into the format that mrs variable is in
  mrs = msviper(mySignature, networks, verbose = F, minsize = Nmin, ges.filter = F)

  regulon = vector(length = length(as.list(names(mrs$es$nes))))
  print(regulon)
  for (r in 1:length(as.list(names(mrs$es$nes)))){
               #regulon[[r]] = strsplit(as.list(names(mrs$es$nes))[[r]],' - ')[[1]][1]
               regulon[r] = strsplit(as.list(names(mrs$es$nes))[[r]],' - ')[[1]][1]
  }
  print("output regulon")
  print(regulon)
  names(mrs$regulon) = regulon
  names(mrs$es$size) = regulon
  names(mrs$es$nes) = regulon
  
  activities = data.frame(regulon = names(mrs$es$nes),#unlist(regulon),
                       activities = mrs$es$nes,
                       size = mrs$es$size[ names(mrs$es$nes) ]
                       #Size_adaptative = sapply(mrs$regulon[ names(mrs$es$nes) ], function(x) sum((x$likelihood/max(x$likelihood))^2) ), # relevant only in weighted regulons
                       #p.value = mrs$es$p.value,
                       #FDR = p.adjust(mrs$es$p.value, method = 'fdr'),
                       )
  rownames(activities) = seq(length=nrow(activities))
  message('')
  return(activities)
}




# --------------  
# -------------- Load and aggregate regulon objects into VIPER Regulon object
# --------------
load_and_merge_regulons = function(regulons_folder, filter_TFs = NULL){
  aggregated_networks = list()
  for (n in network_files){
    message(' - ', n)
    # Load network and fromat regulons
    net = get(load(paste(regulons_folder, '/', n, sep = '')))
    # Clean TF regulon names (such as the Dorothea ones containing the score in their label, example MYC_A)
    net_tfs = sapply(strsplit(names(net), split = ' '), head, 1)
    net_tfs = sapply(strsplit(net_tfs, split = '_'), head, 1)
    # Filter TFs to includo only those perturbed in the experiment
    if (! is.null(filter_TFs))
      net = net[ net_tfs %in% filter_TFs ]
    if( length(net) < 2 ) # At least 2 TF regulons required
      next()
    aggregated_networks = append(aggregated_networks, net)
  }
  return(aggregated_networks)
}




read_desigfile = function(design_file){
  x = t(read.delim(design_file, header = F, sep = '\t'))
  colnames(x) = x[1,]
  my_design = as.list(x[2,])
  my_design$positive_samples = unlist(strsplit(my_design$positive_samples, split = ','))
  my_design$negative_samples = unlist(strsplit(my_design$negative_samples, split = ','))
  return(my_design)
}






###############################
#PROCESSING DATA----------------------
###############################

load(file = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/results/aggregated_activities_formated.rdata')

# Filter regulons to plot for publication
df = subset(df, ! regulon_dataset %in% c('cnio', 'TOP', 'TOP_PANCANCER') )

# group tissue specific regulons
df$regulon_dataset[ df$regulon_group == 'GTEx tissue-specific'] = 'GTEx tissue-specific'
df$regulon_dataset[ df$regulon_group == "GTEx tissue-specific noCOMBAT"] = "GTEx tissue-specific noCOMBAT"
df$regulon_dataset[ df$regulon_group == "cancer-specific"] = "cancer-specific"

# Define test and plot features
balanced_accuracy = F
plot_title = 'B1'
# TFs
TFs = unique(df$TF[ df$is_TF_perturbed ])
num_TFs = length(unique(df$TF[ df$is_TF_perturbed ]))
#num_TFs
write.table(TFs, file = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/benchmark1_TFs.txt', quote = F, row.names = F, col.names = F)


# List differential expression files
signature_files = list.files(path = DE_folder, pattern = 'rdata', full.names = T)
signature_tfs = strsplit(signature_files, '\\.') %>% sapply(., head, 1) %>% strsplit(., '/') %>% sapply(., tail, 1)
signature_ids = strsplit(signature_files, '/') %>% sapply(., tail, 1) %>% gsub('.rdata', '', .)



# Check if TF is differentially expressed as expected
is_TF_DE_expected = mapply(function(tf, fi){
            DEsignature = get(load(fi))
            TF_DEsignature = subset(DEsignature, Symbol == tf)
            any(TF_DEsignature$P.Value < 0.05 & TF_DEsignature$logFC > 0)
          }, signature_tfs, signature_files)
table(is_TF_DE_expected)
which( ! is_TF_DE_expected)



# Check if TF is differentially expressed in an oposite sense
is_TF_DE_opposite = mapply(function(tf, fi){
  DEsignature = get(load(fi))
  TF_DEsignature = subset(DEsignature, Symbol == tf)
  any(TF_DEsignature$P.Value < 0.05 & TF_DEsignature$logFC < 0) & ! any(TF_DEsignature$P.Value < 0.05 & TF_DEsignature$logFC > 0)
}, signature_tfs, signature_files)
table(is_TF_DE_opposite)
which( is_TF_DE_opposite )
is_TF_in_array = mapply(function(tf, fi){
  DEsignature = get(load(fi))
  tf %in% DEsignature$Symbol
}, signature_tfs, signature_files)
table(is_TF_in_array)
which( ! is_TF_in_array)



#source('code/lib/contrast.r')
design_files = list.files(design_folder, recursive = T, pattern = 'G', full.names = T)
designs = lapply(design_files, read_desigfile)
design_id = sapply(designs, function(x) x$id)
design_effect = sapply(designs, function(x) x$effect)
design_treatment = sapply(designs, function(x) x$treatment)
design_GTEx_tissue = sapply(designs, function(x) x$GTEx_tissue)



df = data.frame(id = signature_ids, tf = signature_tfs,
                is_TF_DE_expected = is_TF_DE_expected,
                is_TF_DE_opposite = is_TF_DE_opposite,
                is_TF_in_array = is_TF_in_array,
                perturbation_effect = design_effect[ match(signature_ids, design_id) ],
                perturbation_treatment = design_treatment[ match(signature_ids, design_id) ],
                perturbation_GTExtissue = design_GTEx_tissue[ match(signature_ids, design_id) ],
                stringsAsFactors = F)
write.csv(x = df, file = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/contrasts/TF_perturbation_confidence.csv', row.names = F)



############################
# COMPUTE ACTIVITIES
############################


# Set minimum regulon size
Nmin = 4

# Identify which TFs are perturbed
perturbed_TFs = list.dirs(path = design_folder, full.names = F) %>% setdiff(., '')

message('Merging regulons into one object')
regulons_folderA = paste(regulons_folder,'/omnipath_scores/A',sep="")
regulons_folderB = paste(regulons_folder,'/omnipath_scores/B',sep="")
print("regulons folder A")
print(regulons_folderA)
network_files = list.files(regulons_folderA, recursive = T, pattern = 'viperRegulon.rdata')# %>%
  #grep("specific", ., invert = T, value = T)
print("network files")
print(network_files)
networksA = load_and_merge_regulons(regulons_folderA, filter_TFs = perturbed_TFs)
network_files = list.files(regulons_folderB, recursive = T, pattern = 'viperRegulon.rdata') %>%
  grep("specific", ., invert = T, value = T)
networksB = load_and_merge_regulons(regulons_folderB, filter_TFs = perturbed_TFs)
#network_files = c(network_filesA,network_filesB)
print(networksA)
networks = append(networksA, networksB)
#print(network_files)
#print(networks)

message('Computing differential activities from DE signatures')
perturbation_files = list.files(path = contrast_folder, pattern = 'rdata', full.names = T)

message('pert files')
message(perturbation_files)

for (DE_file in perturbation_files){

  perturbation_id = tail(unlist(strsplit(DE_file, split = '/')),1) %>% gsub('.rdata', '', .) %>% strsplit(., split = '\\.') %>% unlist(.)
  design_file = list.files(design_folder, recursive = T, full.names = T) %>% grep(perturbation_id[1], ., value = T) %>% grep(perturbation_id[2], ., value = T)

  treatment = read_desigfile(design_file)$treatment
  activities = DEs2activities(DE_file, design_folder, networks)
  message("regulon activities")
  message(activities$Regulon)
  analysis_name = gsub('contrasts/', 'activities/global/viper_', DE_file)
  analysis_name = gsub('rdata', 'viper_pred.csv', DE_file)
  print(analysis_name)
  write.csv(activities,analysis_name)
}



