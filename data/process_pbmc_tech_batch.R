#!/usr/bin/env Rscript
library(optparse)
library(Seurat)
library(tidyverse)
library(harmony)

option_list <- list(
  make_option(
    c("-i", "--input-dir"),
    dest = "input_dir",
    type = "character",
    default = NULL,
    help = "path to data directory",
    metavar = "input directory"
  ),
  make_option(
    c("-o", "--output-dir"),
    dest = "output_dir",
    type = "character",
    default = NULL,
    help = "output file directory",
    metavar = "output diirectory"
  )
  
)

read_data <- function(data_dir) {
  # Read raw data
  exprs_raw <- list(
    file.path(data_dir, "fiveprime_raw.rds"),
    file.path(data_dir, "threepv2_raw.rds"),
    file.path(data_dir, "threepv1_raw.rds")
  ) %>%
    lapply(readRDS) %>%
    purrr::reduce(Matrix::cbind2)
  
  # Read metadata
  metadata <- read_csv(file.path(data_dir, "metadata.csv"))
  return(list(expr = exprs_raw, metadata = metadata))
}


process_and_normalise_data = function(expr, metadata) {
  # subset metadata to the two library protocols / technologies
  metadata <- metadata %>%
    filter(donor %in% c("threepfresh", "fivePrime"))
  
  # Prepare metadata for Seurat
  metadata <- as.data.frame(metadata)
  rownames(metadata) <- metadata$cell_id
  # subset data to the two library protocols / technologies
  expr <- expr[, metadata$cell_id]
  # select genes that have at least 1 count across cells
  genes_use <-
    which(Matrix::rowSums(expr[, metadata$cell_id] != 0) > 0)
  # and select genes that aren't mitochondrial
  genes_use <- genes_use[which(!grepl("^MT-", names(genes_use)))]
  # subset genes
  expr <- expr[names(genes_use), ]
  
  pbmc = Seurat::CreateSeuratObject(counts = expr,
                                    meta.data = metadata)
  
  pbmc.list <- SplitObject(pbmc, split.by = "donor")
  pbmc.list <- lapply(
    X = pbmc.list,
    FUN = function(x) {
      x <-
        NormalizeData(x,
                      normalization.method = "LogNormalize",
                      scale.factor = 10000)
      x <-
        FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
    }
  )
  # select features that are repeatedly variable across datasets for integration
  features <- SelectIntegrationFeatures(object.list = pbmc.list)
  pbmc <-
    NormalizeData(pbmc,
                  normalization.method = "LogNormalize",
                  scale.factor = 10000)
  all.genes <- rownames(pbmc)
  pbmc <- ScaleData(pbmc, features = all.genes)
  pbmc <- RunPCA(pbmc, features = features)
  return(
    list(
      seurat_data = pbmc,
      seurat_list = pbmc.list,
      features = features,
      metadata = metadata
    )
  )
}

run_seurat_integration <- function(seurat_list, features) {
  anchors <-
    FindIntegrationAnchors(object.list = seurat_list, anchor.features = features)
  combined <- IntegrateData(anchorset = anchors)
  DefaultAssay(combined) <- "integrated"
  combined <- ScaleData(combined, verbose = FALSE)
  combined <-
    RunPCA(
      combined,
      npcs = 40,
      verbose = FALSE,
      reduction.name = "seurat"
    )
  combined <-
    RunUMAP(
      combined,
      reduction = "seurat",
      dims = 1:40,
      reduction.name = "seurat_umap"
    )
  combined <-
    FindNeighbors(combined, reduction = "seurat", dims = 1:40)
  combined <- FindClusters(combined, resolution = 0.5)
  return(combined)
}


run_harmony <- function(seurat_object) {
  seurat_object <- RunHarmony(seurat_object, "donor")
  seurat_object <- seurat_object %>%
    RunUMAP(reduction = "harmony",
            dims = 1:40,
            reduction.name = "harmony_umap") %>%
    FindNeighbors(reduction = "harmony", dims = 1:40) %>%
    FindClusters(resolution = 0.5) %>%
    identity()
  return(seurat_object)
}


extract_data_for_comp <-
  function(seurat_object, metadata, features) {
    processed_data <- seurat_object@assays$RNA@data[features,] %>%
      as.matrix() %>%
      t()
    
    metadata <- metadata %>%
      mutate(type = donor)
    
    return(list(data = processed_data, metadata = metadata))
  }


opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

if (is.na(opt$input_dir)) {
  stop("Must provide data directory; '--input-dir'")
}

if (is.na(opt$output_dir)) {
  stop("Must provide output directory; '--output-dir'")
}

data_dir <- opt$input_dir
out_dir <- opt$output_dir

# set seed
set.seed(1)

message("Reading data...")
data_list <- read_data(data_dir)
message("Processing data...")
pbmc <-
  process_and_normalise_data(data_list$expr, data_list$metadata)
message("Performing Seurat Integration...")
seurat_integration <-
  run_seurat_integration(pbmc$seurat_list, pbmc$features)
message("Performing Harmony Integration...")
harmony_integration <- run_harmony(pbmc$seurat_data)

seurat_embeddings <- Embeddings(seurat_integration, "seurat")
harmony_embeddings <- Embeddings(harmony_integration, "harmony")
message("Extracting data for CoMP...")
processed_data <-
  extract_data_for_comp(pbmc$seurat_data, pbmc$metadata, pbmc$features)


if (!dir.exists(out_dir)) {
  dir.create(out_dir)
}

message("Writing data and metadata for CoMP to output directory...")
# write processed data and metadata for CoMP
write.table(processed_data$data,
            file = file.path(out_dir, "features.tsv"),
            sep = "\t")
write.table(processed_data$metadata,
            file = file.path(out_dir, "metadata.tsv"),
            sep = "\t")

message("Writing Seurat and Harmony embeddings to output directory...")
# write Seurat and Harmony embeddings
write.table(seurat_embeddings[rownames(processed_data$metadata),],
            file.path(out_dir, "seurat_embeddings.tsv"),
            sep = "\t")
write.table(harmony_embeddings[rownames(processed_data$metadata),],
            file.path(out_dir, "harmony_embeddings.tsv"),
            sep = "\t")
