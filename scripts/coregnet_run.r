#!/usr/bin/env Rscript
library(CoRegNet)
library(jsonlite)
library(reticulate)

# ===== Args =====
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) stop("Please provide EXP_ID as argument.")
EXP_ID <- args[1]
DATASET_ID <- args[2]

# ===== Load Python util =====
util <- import("coregnet_util")

# ===== Start tool =====
cat("Starting experiment:", EXP_ID, "\n")
start_res     <- util$start_tool(EXP_ID,DATASET_ID)
input_details <- start_res[[1]]
dataset       <- start_res[[2]]

# ===== Read coregnet options =====
opts             <- input_details[["exp_details"]][["tool_run"]][["coregnet"]]
threshold        <- if (!is.null(opts[["discretization_threshold"]])) opts[["discretization_threshold"]] else 1
max_coreg        <- if (!is.null(opts[["maxCoreg"]]))                 min(opts[["maxCoreg"]], 5)         else 5
min_coreg_support <- if (!is.null(opts[["minCoregSupport"]]))          opts[["minCoregSupport"]]           else 0.2
min_gene_support  <- if (!is.null(opts[["minGeneSupport"]]))           opts[["minGeneSupport"]]            else 0.1

cat("Options — threshold:", threshold, "| maxCoreg:", max_coreg,
    "| minCoregSupport:", min_coreg_support, "| minGeneSupport:", min_gene_support, "\n")

# ===== Prepare data via Python =====
result     <- util$get_data(input_details, dataset)
r_matrix   <- as.matrix(py_to_r(result[[1]]))
tf_vec     <- py_to_r(result[[2]])
target_vec <- py_to_r(result[[3]])

# ===== Sanity Check =====
cat("--- Sanity Check ---\n")
cat("Matrix dim:", nrow(r_matrix), "genes x", ncol(r_matrix), "samples\n")
cat("TFs:", length(tf_vec), "| Targets:", length(target_vec), "\n")
cat("First 3 rownames:", head(rownames(r_matrix), 3), "\n")
cat("First 3 colnames:", head(colnames(r_matrix), 3), "\n")
cat("Value range: min=", min(r_matrix), "max=", max(r_matrix), "\n")
cat("Any NA:", anyNA(r_matrix), "\n")
cat("TFs in rownames:", sum(tf_vec %in% rownames(r_matrix)), "/", length(tf_vec), "\n")
cat("Targets in rownames:", sum(target_vec %in% rownames(r_matrix)), "/", length(target_vec), "\n")
stopifnot(nrow(r_matrix) == length(tf_vec) + length(target_vec))
stopifnot(!anyNA(r_matrix))
stopifnot(all(tf_vec %in% rownames(r_matrix)))
cat("--- Sanity Check Passed ---\n")

# ===== Discretize =====
cat("Discretizing with threshold:", threshold, "\n")
disc_matrix <- discretizeExpressionData(r_matrix, threshold = threshold)

# ===== Run hLICORN =====
cat("Running hLICORN...\n")
t0  <- proc.time()
grn <- hLICORN(
  numericalExpression = r_matrix,
  discreteExpression  = disc_matrix,
  TFlist              = tf_vec,
  GeneList            = target_vec,
  parallel            = "no",
  maxCoreg            = max_coreg,
  minCoregSupport     = min_coreg_support,
  minGeneSupport      = min_gene_support,
  verbose             = TRUE
)
hlicorn_secs <- round((proc.time() - t0)[["elapsed"]], 2)

# ===== Save results =====
grn_df     <- coregnetToDataframe(grn)
output_dir <- file.path(Sys.getenv("EXP_OUTPUT_PATH"), EXP_ID,DATASET_ID,"coregnet")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
write.csv(grn_df, file.path(output_dir, "grn.csv"), row.names = FALSE)
writeLines(as.character(hlicorn_secs), file.path(output_dir, "runtime.txt"))

cat("Saved to:", output_dir, "\n")
cat("Interactions found:", nrow(grn_df), "\n")
cat("hLICORN runtime:", hlicorn_secs, "seconds\n")

cat("Done.\n")