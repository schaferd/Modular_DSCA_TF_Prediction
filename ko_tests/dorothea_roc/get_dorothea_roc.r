#!/usr/bin/env Rscript

require(pROC)
require(ROCR)
require(PRROC)

library(viper)
library(magrittr)
library(plyr)
library(reshape2)

library(ggplot2)
library(ggrepel)
library(UpSetR)
library(RColorBrewer)
library(cowplot)
library(pheatmap)
library(ggbeeswarm)

my_color_palette = list(EMBL = c('#00777D', '#A2C012', '#b78c00',  '#0081AD' , '#E94949',  brewer.pal(name = 'Dark2', n=8), 'cornflowerblue'),
                        shiny = c('#2B8C7F', '#BFC42C','#C4712D', '#C42D44', '#363636', '#789895',  '#0081AD'),
                        clear = c('#789895' , '#676d7e', '#E94949', '#A2C012', '#DCD6B4', '#E0AD00',  '#0081AD'))


# Define ggplot2 theme
mytheme =   theme_light(15) +
  theme(legend.position = 'bottom', legend.key = element_blank(),
        legend.background = element_rect(fill=alpha('white', 0)),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        plot.background = element_rect(fill=alpha('white', 0)),
        text = element_text(family = 'sans'),
        strip.background =  element_blank(),
        strip.text.x = element_text(color ='black', family = 'sans'), strip.text.y = element_text(color ='black', family = 'sans'),
        legend.title = element_text(size=12),
        legend.text = element_text(size=12),
        axis.text=element_text(size=12, color ='black'),
        axis.title=element_text(size=12, color ='black'),
        plot.title = element_text(size=15, hjust = 0.5))

#ENV VARIABLES
# Set enviroment folders
contrast_folder = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/contrasts'
design_folder = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/design'
regulons_folder ='/nobackup/users/schaferd/ko_eval_data/data/TF_target_sources'
outdir = '/nobackup/users/schaferd/ko_eval_data/results/regulons_QC/B1_perturbations/'
DE_folder = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/contrasts/'




##########################################################################################
#FUNCTIONS###########################
##########################################################################################


# --------------
# -------------- Merge activity_ranks, columns_perturbation_annot and rows_regulons_annot to plot accuracy
# --------------
merge_data2plot_accuracy = function(file){
  load(file)
  mdf = melt(activity_ranks)
  names(mdf) = c('regulon', 'experiment', 'rank_nes')
  mdf$NES = melt(activity_nes)$value
  # mdf$pvalue1tailed = melt(activity_pvalue1tailed)$value
  mdf$perturbed_TF = columns_perturbation_annot$perturbed_TF[ match(mdf$experiment, columns_perturbation_annot$perturbation_id) ]
  mdf$TF = rows_regulons_annot$TF[ match(mdf$regulon, rows_regulons_annot$regulon_id) ]
  mdf$regulon_evidence = rows_regulons_annot$regulon_evidence[ match(mdf$regulon, rows_regulons_annot$regulon_id) ]
  mdf$regulon_dataset = rows_regulons_annot$regulon_dataset[ match(mdf$regulon, rows_regulons_annot$regulon_id) ]
  mdf$regulon_group = rows_regulons_annot$regulon_group[ match(mdf$regulon, rows_regulons_annot$regulon_id) ]
  mdf$is_TF_perturbed = mdf$TF == mdf$perturbed_TF
  mdf$experiment = as.character(mdf$experiment)
  # remove NA activities
  mdf = subset(mdf, ! is.na(NES) )
  return(mdf)
}


# --------------  
# -------------- Manage regulons information
# --------------
manage_datasets_names = function(x, what = 'dataset2group'){
  # message('Options for what are:')
  # message('- regulon_id2TF')
  # message('- regulon_id2regulon_dataset_full')
  # message('- regulon_dataset_full2evidence')
  # message('- regulon_dataset_full2dataset')
  # message('- dataset2group')
  x_out = x
  if(what == 'regulon_id2TF'){
    x_out = sapply(strsplit(x, ' '), head, 1)
    if( length(grep('_[A-E]$', x_out)) > 0 )
      x_out[ grep('_[A-E]$', x_out) ] = sapply(strsplit(x_out[ grep('_[A-E]$', x_out) ], '_'), head, 1)
  } 
  if(what == 'regulon_id2regulon_dataset_full'){
    x_out = sapply(strsplit(x, ' '), tail, 1) %>% gsub('.sif', '', .)
  }
  if(what == 'regulon_dataset_full2evidence'){
    x_out = sapply(strsplit(x, '/'), head, 1)
    x_out[ intersect(grep('PANCANCER', x, ignore.case = T), grep('omnipath', x)) ] = 'omnipath_scores_cancer'
    x_out[ intersect(grep('pancancer', x, ignore.case = T), grep('inferred', x)) ] = 'inferred_cancer'
  }
  if(what == 'regulon_dataset_full2dataset'){
    x_out = sapply(strsplit(x, '/'), function(xx) paste(xx[2:length(xx)], collapse = '_') ) %>%
      gsub('network', '', .) %>%
      gsub('__', '_', .) %>% 
      gsub('_signed_signed', '_signed', .)  
    x_out = sapply(strsplit(x_out, 'via'), head, 1)
    x_out = gsub('_$', '', x_out)  %>% 
      gsub('^_', '', .) 
  }
  if(what == 'dataset2group'){
    x_out[ grep('pantissue', x) ] = 'GTEx pantissue'
    x_out[ intersect(grep('noCOMBAT', x), grep('pantissue', x)) ] = 'GTEx pantissue_noCOMBAT'
    x_out[ grep('pancancer', x) ] = 'pancancer'
    x_out[ grep('inferred ', x) ] = 'GTEx tissue-specific'
    x_out[ intersect(grep('inferred ', x), grep('noCOMBAT', x)) ] = 'GTEx tissue-specific noCOMBAT'
    x_out[ grep('tcga_', x) ] = 'cancer-specific'
    # x_out[ x_out %in% c(LETTERS[1:5], 'TOP') ] = 'omnipath'
    # x_out[ x_out %in% paste(c(LETTERS[1:5], 'TOP'), 'PANCANCER', sep ='_') ] = 'omnipath_cancer'
    x_out[ grep('hocomoco', x_out) ] = 'hocomoco'
    x_out[ grep('ReMap', x_out) ] = 'ReMap'
    x_out[ grep('jaspar', x_out) ] = 'jaspar'
    # x_out[ x_out %in% c(LETTERS[1:5], 'TOP')  ] = 'omnipath_scores'
    x_out = gsub('_signed_signed', '', x_out)
    x_out = gsub('e2_', '', x_out)
  }
  if(what == 'evidence2color'){
    # x_out = c('coral', my_color_palette$EMBL[c(5,5,8,3,4,2,6,1,7)])[
    x_out = c('coral', my_color_palette$EMBL[c(5,5,3,3,4,2,6,1)], brewer.pal(n = 6, name = 'Paired')[6])[
      match(x_out,
            c('ChIP_Seq', 'consensus', 'consensus_all', 'consensus_curated', 'curated_databases', 'inferred',  'old_consensus', 'TFBS_scanning', 'omnipath_scores', 'omnipath_scores_cancer') )]
  }
  if(what == 'evidence2shape'){
    x_out = c(15, 6, 6, 6, 16:17, 5, 18, 16, 16, 1)[
      match(x_out,
            c('ChIP_Seq', 'consensus', 'consensus_all', 'consensus_curated', 'curated_databases', 'inferred',  'old_consensus', 'TFBS_scanning', 'omnipath_scores', 'omnipath_scores_cancer') )]
  }
  # x_out = gsub('_databases', '', x_out) %>% gsub('databases', '', .) %>% gsub('_scanning', '', .) 
  return(x_out)
}



annotate_regulons = function(regulons){
  rows_regulons_annot = data.frame(regulon_id = regulons,
                                   TF = manage_datasets_names(regulons, what = 'regulon_id2TF'),
                                   regulon_dataset_full = manage_datasets_names(regulons, what = 'regulon_id2regulon_dataset_full'),
                                   stringsAsFactors = F)
  rows_regulons_annot$regulon_evidence = manage_datasets_names(rows_regulons_annot$regulon_dataset_full, what = 'regulon_dataset_full2evidence')
  rows_regulons_annot$regulon_dataset = manage_datasets_names(rows_regulons_annot$regulon_dataset_full, what = 'regulon_dataset_full2dataset')
  tissue_specific_index = grep('/', rows_regulons_annot$regulon_dataset_full, invert = T)
  rows_regulons_annot$regulon_dataset[ tissue_specific_index ] =  paste('inferred', rows_regulons_annot$regulon_dataset_full[tissue_specific_index] )
  rows_regulons_annot$regulon_evidence[ tissue_specific_index ] = 'inferred'
  rows_regulons_annot$regulon_group = manage_datasets_names(rows_regulons_annot$regulon_dataset, what = 'dataset2group')
  rows_regulons_annot$regulon_evidence[ rows_regulons_annot$regulon_group == 'cancer-specific' ] = 'inferred_cancer'
  unique(rows_regulons_annot[, c('regulon_dataset', 'regulon_evidence', 'regulon_group') ])
  return(rows_regulons_annot)
}


# --------------
# -------------- Rank activity_nes
# --------------
activity_nes2ranks = function(activity_nes){
  activity_ranks = t(apply(activity_nes, 1, rank, ties.method = 'min'))
  activity_ranks = activity_ranks / apply(activity_ranks, 1, max)
  activity_ranks[ is.na(activity_nes) ] = NA
  return(activity_ranks)
}


# --------------
# -------------- Load and filter differential activities from msVIPER
# --------------
load_activity_msVIPERfile = function(f, Nmin, Nmax){
  # Load
  viper_results = get(load(f))
  # Filter according to the regulon size
  viper_results = subset(viper_results, Size >= Nmin & Size <= Nmax)
  viper_results = viper_results[ order(viper_results$Regulon), ]
  return(viper_results)
}


# --------------
# -------------- Generate the aggregated NES matrix including all regulons
# --------------
aggregate_activities_NESmatrix = function(activities_list){
  #df = melt(activities_list, id.vars = names(activities_list[[1]]))
  
  #message(names(as.list(activities_list[[1]])))
  #activities_list = activities_list[activities_list$name != 'p.value']
  #activities_list = activities_list[activities_list$name != 'p.value']
  #activities_list$p.value <- activities_list$Size_adaptative <- activities_list$FDR <- NULL
  #activities_list = activities_list[c('Regulon','Size','NES','perturbation_tissue','Sample')]
  #for (i in 1:length(activities_list)){
  #	activities_list[[i]] = activities_list[[i]][c('Regulon','Size','NES','perturbation_tissue','Sample')]

  #}
  print("names of activities list")
  message(names(activities_list)[[1]])
  df = melt(activities_list, id.vars = names(activities_list[[1]]))
  print("melted df")
  print(df)
  message("melting")
  print(names(activities_list[[1]]))
  print(summary(df))
  message("df regulon")
  print(df$Regulon)
  message("df sample")
  print(df$Sample)
  df$value = df$NES
  message("df value")
  print(df$value)
  

  #new_df = data.frame(Regulon = df$Regulon, Sample = df$Sample,value=df$value)
  #new_df = melt(new_df, id.vars = names(activities_list[[1]]))
  formula = Regulon ~ Sample
  data = df
  value.var = reshape2:::guess_value(data)
  formula <- reshape2:::parse_formula(formula, names(data), value.var)
  value <- data[[value.var]]
  vars <- lapply(formula, eval.quoted, envir = data, enclos = parent.frame(2))
  drop = TRUE

  print("vars")
  print(vars)
  print("id")
  print(id)
  # Compute labels and id values
  print("length")
  print(length(vars))
  print("ncol")
  print(ncol(vars))
  ids <- lapply(vars, id, drop = drop)
  is_empty <- vapply(ids, length, integer(1)) == 0
  print(is_empty)

  empty <- structure(rep(1, nrow(data)), n = 1L)
  ids[is_empty] <- rep(list(empty), sum(is_empty))

  labels <- mapply(plyr:::split_labels, vars, ids, MoreArgs = list(drop = drop),
    SIMPLIFY = FALSE, USE.NAMES = FALSE)
  labels[is_empty] <- rep(list(data.frame(. = ".")), sum(is_empty))

  print("func ids")
  print(ids)

  overall <- id(rev(ids), drop = FALSE)
  n <- attr(overall, "n")

  fill = NA


  # ordered <- vaggregate(.value = value, .group = overall, .fun = fun.aggregate, ...,  .default = fill, .n = n)
  .value = value
  .group = overall
  .fun = NULL 
  .default = fill
  .n = n

  print(".group 1")
  print(.group)

  if (!is.integer(.group)) {
    if (is.list(.group)) {
      .group <- id(.group)
    } else {
      .group <- id(list(.group))
    }
  }

  if (is.null(.default)) {
    .default <- .fun(.value[0], ...)
  }

  fun <- function(i) {
    if (length(i) == 0) return(.default)
    .fun(.value[i], ...)
  }

  print(".group")
  print(.group)

  print(".n")
  print(.n)

  indices <- split_indices(.group, .n)

  print("indices")
  print(indices)
  print("fun")
  print(fun)
  print(".default")
  print(.default)




  activity_nes = acast(df, Regulon ~ Sample, fill = NA)
  print("tilde output")
  print(activity_nes)
  return(activity_nes)
}

# --------------COMPUTE TF ACTIVITIES FUNCTION----------------------------------------
# -------------- Compute TF activities from DE signatures
# -------------- NEED TO FIGURE OUT HOW TO IMPORT FUNCTION MSVIPER


get_aucPR = function(observed, expected){
  aucPR = pr.curve(scores.class0 = observed[ expected == 0], scores.class1 = observed[ expected == 1])$auc.integral
  return(aucPR)
}


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
  print("hello there")
  #regulon = names(mrs$es$nes)
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
  names(mrs$es$nes) = regulon
  print("tissue names")
  print(tissue)
  
  activities = data.frame(Regulon = names(mrs$es$nes),#unlist(regulon),
                       Size = mrs$es$size[ names(mrs$es$nes) ],
                       #Size_adaptative = sapply(mrs$regulon[ names(mrs$es$nes) ], function(x) sum((x$likelihood/max(x$likelihood))^2) ), # relevant only in weighted regulons
                       NES = mrs$es$nes,
                       #p.value = mrs$es$p.value,
                       #FDR = p.adjust(mrs$es$p.value, method = 'fdr'),
                       perturbation_tissue = tissue)

  
  print("perturbation tissue")
  print(length(activities$perturbation_tissue))
  print("NES")
  print(length(activities$NES))
  print("size")
  print(length(activities$Size))
  print("regulon")
  print(length(activities$Regulon))

  #message('activities: Size_adaptative = sapply(mrs$regulon[ names(mrs$es$nes) ], function(x) sum((x$likelihood/max(x$likelihood))^2) ), # relevant only in weighted regulons')
  #message(sapply(mrs$regulon[ names(mrs$es$nes) ], function(x) sum((x$likelihood/max(x$likelihood))^2)))

  #message('activities: NES = mrs$es$nes')
  #message(mrs$es$nes)

  #message('activities: p.value = mrs$es$p.value')
  #message(mrs$es$p.value)

  #message("activities: FDR = p.adjust(mrs$es$p.value, method = 'fdr'")
  #message(p.adjust(mrs$es$p.value, method = 'fdr'))

  activities = activities[ order(activities$NES, decreasing = T) , ]
  message('')
  return(activities)
}




# --------------  
# -------------- Load and aggregate regulon objects into VIPER Regulon object
# --------------
load_and_merge_regulons = function(regulon_files, regulons_folder, filter_TFs = NULL){
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





#--------------COMPUTE ACCURACY FROM ACTIVITIES------------------------
activities2accuracy = function(ddf, y='NES', wrap_variable='regulon_dataset',
                             balanced_accuracy = F, performance_method = 'aucPR',
                             regulon_evidence_filter=NULL){
  ddf$wrap_variable = ddf[, wrap_variable]
  ddf$y = ddf[, y]
  if( ! is.null(regulon_evidence_filter) )
    ddf = subset(ddf, regulon_evidence %in% regulon_evidence_filter)
  wrap_variable_filter = names(which(table(unique(ddf[, c('wrap_variable', 'is_TF_perturbed') ])$wrap_variable) == 2))
  ddf = subset(ddf, wrap_variable %in% wrap_variable_filter)
  # Compute performance ~ AUC
  df_coverage2accuracy = ddply(ddf, c(wrap_variable, 'regulon_evidence', 'regulon_dataset'), function(x) {
    observed = x$y
    expected = (x$is_TF_perturbed + 0)
    performance_value = compute_accuracy(observed, expected, method = performance_method, balanced = balanced_accuracy)
    return(performance_value)
  })
  # Format dataframe
  names(df_coverage2accuracy)[ names(df_coverage2accuracy) == 'V1' ]  = 'AUC'
  df_coverage2accuracy$coverage = ddply(ddf, c(wrap_variable, 'regulon_evidence', 'regulon_dataset'), function(x) {
    length(unique(x$TF[ x$is_TF_perturbed ]))
  })$V1
  # df_coverage2accuracy$coverage[ grep('inferredGTEx ', df_coverage2accuracy$regulon_dataset, ignore.case = T) ] = length(unique(ddf$TF[ intersect(which(ddf$is_TF_perturbed), grep('inferredGTEX ', ddf$regulon_dataset, ignore.case = T) ) ])) # aggregate coverage from tissue-specific networks
  df_coverage2accuracy = df_coverage2accuracy[ order(df_coverage2accuracy$AUC, decreasing = T), ]
  return(df_coverage2accuracy)
}





#-----------COMPUTE ACCURACY FUNCTION----------------------------------
compute_accuracy = function(observed, expected, method = 'aucPR', balanced = F){
  if(balanced){
    if( method == 'aucROC'){
      performance_value = get_aucROC(observed, expected)
    }
    if( method == 'aucPR'){
      performance_value = get_aucPR(observed, expected)
    }
  }else{
    n_positives = sum(expected==1)
    n_negatives = sum(expected==0)
    positives = observed[ expected == 1 ]
    negatives = observed[ expected == 0 ]
    n = min(n_positives, n_negatives)
    r_negatives = lapply(1:100, function(ra) sample(negatives, n, replace = F)  ) # down-sample the negatives to balance
    r_positives = lapply(1:100, function(ra) sample(positives, n, replace = F)  ) # down-sample the positives to balance
    if( method == 'aucROC'){
      ac = mapply(function(ne, po) get_aucROC(c(ne, po),  c(rep(0, n), rep(1, n) ) ), r_negatives, r_positives)
    }
    if( method == 'aucPR'){
      # ac = sapply(r_negatives, function(ne) get_aucPR(c(ne, positives),  c(rep(0, n_positives), rep(1, n_positives) ) )  )
      ac = mapply(function(ne, po) get_aucPR(c(ne, po),  c(rep(0, n), rep(1, n) ) ), r_negatives, r_positives)
    }
    performance_value = mean(unlist(ac))
  }
  return(performance_value)
}




#--------------------GET AUC ROC FUNCTION-------------------------------------
get_aucROC = function(observed, expected){
  myroc = roc(predictor = observed, response = expected, smooth=F)
  performance_value = myroc$auc[1]
  return(performance_value)
}



plot_rocs = function(ddf, y='NES', wrap_variable='regulon_dataset',
                   balanced_accuracy = F,
                   regulon_evidence_filter=NULL, TFs_filter = NULL,
                   line_colors = my_color_palette$EMBL){
  ddf$wrap_variable = ddf[, wrap_variable]
  ddf$y = ddf[, y]
  if( ! is.null(regulon_evidence_filter) )
    ddf = subset(ddf, regulon_evidence %in% regulon_evidence_filter)
  if( ! is.null(TFs_filter) )
    ddf = subset(ddf, TF %in% TFs_filter)
  if( length(line_colors) == 1)
    line_colors = rep(line_colors, 100)
  wrap_variable_filter = names(which(table(unique(ddf[, c('wrap_variable', 'is_TF_perturbed') ])$wrap_variable) == 2))
  ddf = subset(ddf, wrap_variable %in% wrap_variable_filter)
  message(ddf)
  df_accuracy = dlply(ddf, c(wrap_variable, 'regulon_evidence'), function(x) {
    observed = x$y
    expected = (x$is_TF_perturbed + 0)
    if(balanced_accuracy){
      auc = get_aucROC(observed, expected)
    }else{
      n_positives = sum(expected==1)
      n_negatives = sum(expected==0)
      positives = observed[ expected == 1 ]
      negatives = observed[ expected == 0 ]
      n = min(n_positives, n_negatives)
      r_negatives = lapply(1:100, function(ra) sample(negatives, n, replace = F)  ) # down-sample the negatives to balance
      r_positives = lapply(1:100, function(ra) sample(positives, n, replace = F)  ) # down-sample the positives to balance
      auc = mapply(function(ne, po) get_aucPR(c(ne, po),  c(rep(0, n), rep(1, n) ) ), r_negatives, r_positives)
    }
    return(auc)
  })
  message(df_accuracy)
  df_accuracy = melt(df_accuracy)
  message('roc')
  message(df_accuracy$L1)
  df_accuracy[[wrap_variable]] = sapply(strsplit(df_accuracy$L1, split = "\\."), head, 1)
  df_accuracy$regulon_evidence = sapply(strsplit(df_accuracy$L1, split = "\\."), tail, 1)
  df_accuracy$wrap_variable = df_accuracy[, wrap_variable]

  P = ggplot(df_accuracy, aes(x=wrap_variable, y=value, color = wrap_variable)) +
    geom_hline(yintercept = 0.5) +
    geom_boxplot(fill = 'white', alpha = 0, outlier.size = NA, outlier.color = NA, outlier.shape = NA) +
    geom_quasirandom(size = 0.75, alpha =.3) +
    scale_y_continuous(limits = c(0.4, 1) ) +
    coord_flip() +
    scale_color_manual(values = line_colors, name = '') +
    geom_abline(intercept=1, slope=1, linetype="dashed") + xlab('') + ylab("Area under the ROC curve (AUC)") +
    mytheme + theme(legend.position = 'none')
  if( ! is.null(TFs_filter) )
    P = P + annotate('text', x = Inf, y = Inf, label = paste('n =', length(unique(ddf$TF))), vjust = 1.2, hjust = 1.2)
  P
  return(P)
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

#TO DO:
#UNDERSTAND DF STRUCTURE OF OUTPUT OF VIPER 
#determine what parts of viper output are used
#MIMIC DF STRUCTURE WHEN OUTPUTTING OF MY ENCODER

# Set minimum regulon size
Nmin = 4

# Identify which TFs are perturbed
perturbed_TFs = list.dirs(path = design_folder, full.names = F) %>% setdiff(., '')

message('Merging regulons into one object')
network_files = list.files(regulons_folder, recursive = T, pattern = 'viperRegulon.rdata') %>%
  grep("specific", ., invert = T, value = T)
networks = load_and_merge_regulons(regulon_files, regulons_folder, filter_TFs = perturbed_TFs)

message('Computing differential activities from DE signatures')
perturbation_files = list.files(path = contrast_folder, pattern = 'rdata', full.names = T)

message('pert files')
message(perturbation_files)

for (DE_file in perturbation_files){
  activities = DEs2activities(DE_file, design_folder, networks)
  message("regulon activities")
  message(activities$Regulon)
  analysis_name = gsub('contrasts/', 'activities/global/viper_', DE_file)
  save(activities, file = analysis_name)
}


# Set regulon size thresolds
Nmax = 25000
Nmin = 5


# List activity files
activity_files = list.files(path = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/activities/global', pattern = 'rdata', full.names = T, recursive = T)
message("activity files")
message(activity_files)
# Load activities data
activities_list = list()
for (f in activity_files){
  df = load_activity_msVIPERfile(f, Nmin, Nmax)
  print("load activity msviper file")
  print(df)
  message("regulon activities 2")
  print(df$Regulon)
  experiment = strsplit(f, '/') %>% unlist(.) %>% tail(., 1) %>%
    gsub('.rdata', '', .) %>% strsplit(., '_') %>% unlist(.) %>% tail(., 1)
  df$Sample = experiment
  if( length(grep('noCOMBAT', f)) == 1 ){ #noCOMBAT analysis requested by a referee. Ignore it otherwise
    df$Regulon = paste( df$Regulon, '_noCOMBAT', sep ='') # This adds the "_noCOMBAT" label to the regulon name. 
    df$perturbation_tissue = paste( df$perturbation_tissue, '_noCOMBAT', sep ='')
  }
  activities_list[[f]] = df
}
message("activities list 1")
print(names(activities_list[[1]]))
print(activities_list)

message("aggregating activities")
# Generate activities matrixes integrating all the regulons
activity_nes = aggregate_activities_NESmatrix(activities_list)

message("generating ranks")
# Generate ranks
activity_ranks = activity_nes2ranks(activity_nes)


message("annotating regulons")
# Regulons information - ROWS
regulons = rownames(activity_nes)
message("regulons for annotation")
message(regulons)
rows_regulons_annot = annotate_regulons(regulons)
length(regulons)

#save(regulons, file = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/contrasts/regulons_for_annotation.rdata')

message("perturbation var")
# Perturbation information - COLUMNS
perturbation = colnames(activity_nes)
columns_perturbation_annot = data.frame(perturbation_id = perturbation,
                                perturbed_TF = sapply(strsplit(perturbation, '\\.'), head, 1),
                                perturbation_GEOid = strsplit(perturbation, '\\.') %>% sapply(., tail, 1) %>% strsplit(., '-') %>%  sapply(., head, 1),
                                stringsAsFactors = F)
message("saving data")

# Save data
save(activity_nes, 
     activity_ranks,
     columns_perturbation_annot, rows_regulons_annot,
     file = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/results/aggregated_activities.rdata')


message("running data2plot")
# Prepare data frame to plot accuracy
df = merge_data2plot_accuracy(file = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/results/aggregated_activities.rdata')
message("complete merge_data2plot")
message("regulon evidence")
message(df$regulon_evidence)


# Define perturbed TFs
df$is_TF_perturbed = df$TF == df$perturbed_TF


# add experiment information
experiments_info = read.csv('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/contrasts/TF_perturbation_confidence.csv', stringsAsFactors = F)
experiments_info$perturbation_GEOid = strsplit(as.character(experiments_info$id), '\\.') %>% sapply(., tail, 1) %>% strsplit(., '-') %>%  sapply(., head, 1)
df$perturbation_effect = experiments_info$perturbation_effect[ match(df$experiment, experiments_info$id) ]
df$perturbation_treatment = experiments_info$perturbation_treatment[ match(df$experiment, experiments_info$id) ]
df$perturbation_GEOid = experiments_info$perturbation_GEOid[ match(df$experiment, experiments_info$id) ]
# NOTE: we checked if the TF expression changed according the perturbation (i.e. higher expression after overexpression & lower expression after inhibition)
df$is_TF_DE_opposite = experiments_info$is_TF_DE_opposite[ match(df$experiment, experiments_info$id) ] # means the TF is differentially expressed in the opposite direction as expected
df$is_TF_DE_expected = experiments_info$is_TF_DE_expected[ match(df$experiment, experiments_info$id) ] # means the TF is differentially expressed as expected

df = subset(df, ! is_TF_DE_opposite)
df = subset(df, is_TF_DE_expected | ! perturbation_treatment %in% c("shRNA", "siRNA", "overexpression" ) )
tfs_of_interest =  intersect(df$TF , df$perturbed_TF)
message("perturbed tf")
message(df$perturbed_TF)
message("tfs")
message(df$TF)
message('tfs of interest')
message(tfs_of_interest)
df = subset(df, TF %in% tfs_of_interest & perturbed_TF %in% tfs_of_interest) # Filter TFs: only test TFs that are perturbed at least in one experiment
# Invert rank
df$rank_nes = 1 - df$rank_nes
df$NES = df$NES * -(1)
# save
save(df, file = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/results/aggregated_activities_formated.rdata')


# Load data
load(file = '/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B1_perturbations/results/aggregated_activities_formated.rdata')


# Filter regulons to plot for publication
df = subset(df, ! regulon_dataset %in% c('cnio', 'TOP', 'TOP_PANCANCER') )

balanced_accuracy = F
plot_title = 'B1'
outdir = '/nobackup/users/schaferd/ko_eval_data/results/regulons_QC/B1_perturbations/'

sdf = df[ df$regulon_evidence == 'curated_databases',]
message("sdf")
message(sdf)
message("plot rocs")
p = plot_rocs(sdf,
              balanced_accuracy = balanced_accuracy,
              line_colors =  my_color_palette$EMBL[3])

ggsave(filename='roc_plots_dorothea.pdf',plot=p,path=outdir)
