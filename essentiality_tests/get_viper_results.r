source('code/lib/utils.r')

# Set folders
expression_signature_cell_lines = 'data/regulons_QC/B3_cell_lines/expression/voom_batchcor_duplicates_merged.RData'
act_folder = 'data/regulons_QC/B3_cell_lines/activities/'
regulons_folder = 'data/TF_target_sources/'




# Load expression data and compute zscores
message('\n\nLoading zscore-transformed gene expression data')
E_cell_lines = load(expression_signature_cell_lines) %>% get(.) %>% t(.) %>% scale(.) %>% t(.)
# Explude samples without COSMIC id since we do not have phenotypic data for these
load('data/regulons_QC/B3_cell_lines/cellines2tissues_mapping.rdata')
E_cell_lines = E_cell_lines[, colnames(E_cell_lines) %in% celllines_annotation$COSMIC_ID]




# Set min gene set size
N = 4



# Compute activities for each regulon dataset
message('Computing activities')
networks = list.files(regulons_folder, recursive = T, pattern = 'viperRegulon.rdata')
for (reg in networks ){
  message(' - ', reg)
  # Format regulon-specific activities outfile
  act_file = reg %>% gsub('/', '.', .) %>% gsub('viperRegulon_', '', .) %>%  gsub('viperRegulon', '', .) %>% 
    gsub('sif', 'activities.rdata', .)  %>% gsub('\\.\\.', '.', .)  %>% gsub('_\\.', '.', .)
  # Load regulons
  regulon = load(paste(regulons_folder, reg, sep = '')) %>% get(.)
  ## Cell lines activities
  activities = viper(eset = E_cell_lines, regulon = regulon, minsize = N, nes = T, method = 'none', eset.filter = F, pleiotropy = F, verbose = F)
  activities = cbind(Size = sapply(rownames(activities), function(tf) round(sum(regulon[[tf]]$likelihood), digits = 0)  ) , activities)
  save(activities, file = paste(act_folder, act_file, sep = ''))
  write.table(activities,file=paste(out_dir,n_outfile,sep='/'))
}
