#!/usr/bin/env Rscript
# fetch_tcga.R
#
# Reads datasets.json, finds all entries with file_output = "tcga_counts",
# checks if already downloaded, and fetches any that are missing.
#
# Usage (direct):
#   Rscript scripts/fetch_tcga.R
#
# Usage (via make):
#   make tcga-datasets

suppressPackageStartupMessages({
  library(TCGAbiolinks)
  library(SummarizedExperiment)
  library(arrow)
  library(jsonlite)
})

# ─── Config ───────────────────────────────────────────────────────────────────

data_path <- Sys.getenv("DATA_PATH")
if (nchar(data_path) == 0) stop("DATA_PATH env var not set")

# datasets.json lives next to this script's parent directory
datasets_json <- file.path("/app", "datasets.json")
if (!file.exists(datasets_json)) stop("datasets.json not found at: ", datasets_json)


# ─── Helpers ──────────────────────────────────────────────────────────────────

is_downloaded <- function(dataset_dir) {
  file.exists(file.path(dataset_dir, "counts_raw.parquet")) &&
  file.exists(file.path(dataset_dir, "metadata.parquet"))
}


fetch_tcga_dataset <- function(dataset, data_path) {
  project    <- dataset$project
  dataset_id <- dataset$id
  out_dir    <- file.path(data_path, dataset_id)

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  message("\n", strrep("=", 60))
  message("  Fetching: ", project, "  →  ", out_dir)
  message(strrep("=", 60))

  # 1. Query ───────────────────────────────────────────────────────────────────
  message("\n[1/3] Querying GDC ...")
  query <- GDCquery(
    project       = project,
    data.category = "Transcriptome Profiling",
    data.type     = "Gene Expression Quantification",
    workflow.type = "STAR - Counts"
  )

  # 2. Download ────────────────────────────────────────────────────────────────
  message("[2/3] Downloading files ...")
  GDCdownload(
    query,
    method          = "api",
    files.per.chunk = 100    # avoids GDC connection drops on large projects
  )

  # 3. Prepare matrix ──────────────────────────────────────────────────────────
  message("[3/3] Preparing counts matrix ...")
  se <- GDCprepare(query)

  # Raw unstranded counts (required for DESeq2)
  counts <- as.data.frame(assay(se, "unstranded"))

  # Map Ensembl IDs → HGNC gene names
  gene_info        <- as.data.frame(rowData(se))
  counts$gene_name <- gene_info$gene_name
  counts           <- counts[!duplicated(counts$gene_name) & !is.na(counts$gene_name), ]
  rownames(counts) <- counts$gene_name
  counts$gene_name <- NULL

  # Transpose → samples × genes  (matches GTEx convention)
  counts_t <- as.data.frame(t(counts))

  # ── Metadata ──────────────────────────────────────────────────────────────
  col_data   <- as.data.frame(colData(se))
  keep_cols  <- intersect(
    c("barcode", "sample_type", "gender", "tumor_stage", "definition",
      "primary_diagnosis", "age_at_index", "vital_status"),
    colnames(col_data)
  )
  meta          <- col_data[, keep_cols, drop = FALSE]
  meta$is_tumor <- grepl("tumor", tolower(meta$sample_type))

  # ── Save parquet ──────────────────────────────────────────────────────────
  arrow::write_parquet(counts_t, file.path(out_dir, "counts_raw.parquet"))
  arrow::write_parquet(meta,     file.path(out_dir, "metadata.parquet"))

  # ── Save metadata.json (same convention as GTEx datasets) ─────────────────
  meta_json <- list(
    id             = dataset_id,
    title          = paste(project, "Gene Expression Raw Counts"),
    source         = "GDC / TCGA",
    project        = project,
    file_name      = "counts_raw.parquet",
    file_output    = "tcga_counts",
    about          = paste0(
      "STAR raw counts from GDC for project ", project, ". ",
      "Rows=samples, Cols=HGNC gene names. ",
      "Values=unstranded raw integer counts (suitable for DESeq2)."
    ),
    samples        = nrow(counts_t),
    genes          = ncol(counts_t),
    tumor_samples  = sum(meta$is_tumor),
    normal_samples = sum(!meta$is_tumor)
  )
  write_json(meta_json, file.path(out_dir, "metadata.json"),
             pretty = TRUE, auto_unbox = TRUE)

  message("\n✓ Done: ", project)
  message("  samples=", nrow(counts_t),
          "  tumor=",   sum(meta$is_tumor),
          "  normal=",  sum(!meta$is_tumor))
  message("  saved → ", out_dir)
}


# ─── Main ─────────────────────────────────────────────────────────────────────

catalogue <- fromJSON(datasets_json)$datasets

# Filter to only tcga_counts entries
tcga_datasets <- catalogue[catalogue$file_output == "tcga_counts", ]

if (nrow(tcga_datasets) == 0) {
  message("No datasets with file_output='tcga_counts' found in datasets.json")
  quit(status = 0)
}

message("Found ", nrow(tcga_datasets), " TCGA dataset(s) in datasets.json")

ok     <- 0
failed <- c()

for (i in seq_len(nrow(tcga_datasets))) {
  ds         <- as.list(tcga_datasets[i, ])
  dataset_id <- ds$id
  dataset_dir <- file.path(data_path, dataset_id)

  if (is_downloaded(dataset_dir)) {
    message("\n[skip] ", ds$project, " already downloaded at ", dataset_dir)
    ok <- ok + 1
    next
  }

  tryCatch({
    fetch_tcga_dataset(ds, data_path)
    ok <- ok + 1
  }, error = function(e) {
    message("\n[failed] ", ds$project, ": ", conditionMessage(e))
    failed <<- c(failed, ds$project)
  })
}

message("\n", strrep("─", 60))
message(ok, "/", nrow(tcga_datasets), " successful")
if (length(failed) > 0) {
  message("failed: ", paste(failed, collapse = ", "))
}