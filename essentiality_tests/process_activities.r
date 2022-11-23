
process_activities = function(activities_dir) {
	activity_files = list.files(activities_dir,full.names=T,pattern='csv')

	celllines_annotation = load('/nobackup/users/schaferd/ko_eval_data/data/regulons_QC/B3_cell_lines/cellines2tissues_mapping.rdata')
	GTEx_labels = setdiff(celllines_annotation$GTEx, 'none')
	tcga_labels = setdiff(celllines_annotation$Study.Abbreviation, '')

	activities_list = list()
	for (f in activity_files){
	  df = read.csv(f) #load_activity_VIPERfile(f, Nmin, Nmax)
	  if(is.null(df))
	    next
	  f_name = tail(unlist(strsplit(f,'/')),n=1)
	  tissue = intersect(c(GTEx_labels, tolower(tcga_labels)), unlist(strsplit(f_name,'\\.')))
	  if( length(tissue) == 1 ){ #  if tissue-specific regulon; then match regulon to cell-type
	    which_samples_in_tissue = which(celllines_annotation$GTEx %in% tissue |
					    celllines_annotation$Study.Abbreviation %in% toupper(tissue))
	    ###################################
	    cell_lines_in_tissue =  celllines_annotation$COSMIC_ID[ which_samples_in_tissue ]
	    cell_lines_in_tissue =  setdiff(cell_lines_in_tissue, NA) # Remove samples without cosmic id since we do not have phenotypic data for these
	    df = subset(df, Sample %in% cell_lines_in_tissue  ) # remove samples not matching the regulon tissue-type
	  }
	  activities_list[[f]] = df
	}

	activity_nes = aggregate_activities_NESmatrix(activities_list)
	activity_ranks = activity_nes2ranks(activity_nes)

	df = merge_data2plot_accuracy(activity_ranks)

	essentiality = load_ACHILLES()
	essentiality = t(scale(t(essentiality)))
	essentiality = essentiality[ rownames(essentiality) %in% df$TF, ]
	df$TF_essentiality_achilles = NA
	idx = df$TF %in% rownames(essentiality)  & df$experiment %in% colnames(essentiality)
	df$TF_essentiality_achilles[idx] = essentiality[ cbind(df$TF[idx], df$experiment[idx]) ]

	essentiality = load_DRIVEproject()
	essentiality = t(scale(t(essentiality)))
	essentiality = essentiality[ rownames(essentiality) %in% df$TF, ]
	df$TF_essentiality_DRIVEproject = NA
	idx = df$TF %in% rownames(essentiality)  & df$experiment %in% colnames(essentiality)
	df$TF_essentiality_DRIVEproject[idx] = essentiality[ cbind(df$TF[idx], df$experiment[idx]) ]

	# Define essential genes
	df$is_TF_essential = F
	df$is_TF_essential[ which(df$TF_essentiality_achilles < -4 | df$TF_essentiality_DRIVEproject < -4 ) ] = T
	df$is_TF_nonessential = F
	df$is_TF_nonessential[  which(df$TF_essentiality_achilles > 4 | df$TF_essentiality_DRIVEproject > 4 ) ] = T

	# Add homoDeletions
	cell_lines_homoDel = load_homDel()
	cell_lines_homoDel = cell_lines_homoDel[ rownames(cell_lines_homoDel) %in% df$TF, ]
	df$is_TF_deleted = NA
	idx = df$TF %in% rownames(cell_lines_homoDel)  & df$experiment %in% colnames(cell_lines_homoDel)
	df$is_TF_deleted[idx] = cell_lines_homoDel[ cbind(df$TF[idx], df$experiment[idx]) ] == 1

	# Define positive and negative samples
	df$is_TF_inactive = F
	df$is_TF_active = F
	df$is_TF_inactive[ which(df$is_TF_deleted) ] = T
	df$is_TF_inactive[ which(df$is_TF_nonessential) ] = T
	df$is_TF_active[ which(df$is_TF_essential) ] = T

	df$is_TF_perturbed = df$is_TF_active

	# Filter TFs of interest: i.e. TFs in the active/inactive group
	df$is_TF_of_interest = F
	df$is_TF_of_interest [ df$is_TF_inactive | df$is_TF_active ] = T
	df = subset(df, is_TF_of_interest)

	# Invert rank
	df$rank_nes = 1 - df$rank_nes
	df$NES = df$NES * -(1)

	write.table(activity_nes,file=paste(activities_dir,'activities.csv',sep='/'))
	write.table(activity_ranks,file=paste(activities_dir,'activity_ranks.csv',sep='/'))
}


aggregate_activities_NESmatrix = function(activities_list){
  df = melt(activities_list, id.vars = names(activities_list[[1]]))
  df$value = df$NES
  activity_nes = acast(df, Regulon ~ Sample, fill = NA)
  return(activity_nes)
}

activity_nes2ranks = function(activity_nes){
  activity_ranks = t(apply(activity_nes, 1, rank, ties.method = 'min'))
  activity_ranks = activity_ranks / apply(activity_ranks, 1, max)
  activity_ranks[ is.na(activity_nes) ] = NA
  return(activity_ranks)
}

merge_data2plot_accuracy = function(activity_ranks){
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

