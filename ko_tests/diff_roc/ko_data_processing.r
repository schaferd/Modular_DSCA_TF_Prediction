read_desigfile = function(design_file){
  x = t(read.delim(design_file, header = F, sep = '\t'))
  colnames(x) = x[1,]
  my_design = as.list(x[2,])
  my_design$positive_samples = unlist(strsplit(my_design$positive_samples, split = ','))
  my_design$negative_samples = unlist(strsplit(my_design$negative_samples, split = ','))
  return(my_design)
}

design_to_avgExp = function(sample,genes,exp,des,design_file,out_dir,tf){
  p_df = data.frame(genes)
  p_df$avg = rowMeans(sample[,des$positive_samples])
  
  n_df = data.frame(genes)
  n_df$avg = rowMeans(sample[,des$negative_samples])
  print(head(n_df))
  
  n_outfile = paste(tf,gsub('.txt','',design_file),'negative.csv',sep='_')
  write.table(n_df,file=paste(out_dir,n_outfile,sep='/'))
  
  p_outfile = paste(tf,gsub('.txt','',design_file),'positive.csv',sep='_')
  write.table(p_df,file=paste(out_dir,p_outfile,sep='/'))
}

get_output_files = function(design_dir,norm_dir,out_dir){
  tfs = list.dirs(design_dir)[-1]
  for (tf_dir in tfs){
    tf = tail(unlist(strsplit(tf_dir,split='/')),1)
    exps = list.files(tf_dir)
    for (exp_file in exps){
      result = tryCatch({
          id = substr(exp_file,0,3)
          print(id)
          des = read_desigfile(paste(design_dir,tf,exp_file,sep='/'))
          exp = get(load(paste(norm_dir,gsub('txt','rdata',exp_file),sep='/')))
          if (id == "GSE"){
            #genes = exp$GEOobj@gpls$GPL570@dataTable@table$ENTREZ_GENE_ID
            #genes = exp$GEOobj@gpls$GPL570@dataTable@table$`Gene Symbol`
            genes = exp$GEOobj@gpls$GPL96@dataTable@table$`Gene Symbol`
            sample = exp$eset@assayData$exprs
            design_to_avgExp(sample,genes,exp,des,exp_file,out_dir,tf)
          }
          else{
            genes = exp$eset@featureData@data$Symbol
            sample = exp$eset@assayData$exprs
            design_to_avgExp(sample,genes,exp,des,exp_file,out_dir,tf)
          }
          #design_to_avgExp(design_dir,tf,exp,norm_dir,out_dir)
      },error = function(err){
        print("different format")
        },
      finally ={
        print(exp_file)
      })
    }
    
  }
}

norm_dir = '/home/bagel/Documents/ae_data/normalized/'
design_dir = '/home/bagel/Documents/ae_data/design/'
design_tf = 'AR'
design_file = 'GSE11428.txt'
out_dir = '/home/bagel/Documents/ae_out/'
#design_to_avgExp(design_dir,design_tf,design_file,norm_dir,out_dir)
get_output_files(design_dir,norm_dir,out_dir)

