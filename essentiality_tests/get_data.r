library(magrittr)
library(dplyr)

outdir = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B3_cell_lines/expression/'
data_outfile = 'voom_batchcor_duplicates_merged.csv'
expression_signature_cell_lines = paste(outdir,'/voom_batchcor_duplicates_merged.RData',sep='')

E_cell_lines = load(expression_signature_cell_lines) %>% get(.) %>% t(.) %>% scale(.) %>% t(.)
# Explude samples without COSMIC id since we do not have phenotypic data for these
load('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B3_cell_lines/cellines2tissues_mapping.rdata')
E_cell_lines = E_cell_lines[, colnames(E_cell_lines) %in% celllines_annotation$COSMIC_ID]
print(colnames(E_cell_lines))
write.table(E_cell_lines,file=paste(outdir,data_outfile,sep=''),sep=',')
