# DGE.py
from exp_init import load_tflist, resolve_gene_lists
from util import get_analysis_output_path, get_analysis_input_path, get_dataset

import os
import logging
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA


#  Logging setup 

def get_logger(output_path: Path) -> logging.Logger:
    """
    Returns a logger that writes to both console and a run.log file.
    If run.log already exists (rerun), it appends — so the full history is preserved.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "run.log"

    logger = logging.getLogger(str(output_path))  # unique logger per output path
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if logger is reused in the same session
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File handler — always append so reruns add to the same log
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


#  Input generation 

def generate_input(options: dict, log: logging.Logger) -> ad.AnnData:
    _, disease_count = get_dataset(options.get('disease_dataset'),normalize=False)
    _, normal_count  = get_dataset(options.get('normal_dataset'),normalize=False)

    # Check and remove duplicate gene columns in raw data
    n_dupe_disease = disease_count.columns.duplicated().sum()
    n_dupe_normal  = normal_count.columns.duplicated().sum()
    log.info(f"Raw duplicate gene columns — disease: {n_dupe_disease}, normal: {n_dupe_normal}")

    disease_count = disease_count.loc[:, ~disease_count.columns.duplicated(keep='first')]
    normal_count  = normal_count.loc[:,  ~normal_count.columns.duplicated(keep='first')]

    log.info(f"After dedup — disease shape: {disease_count.shape}, normal shape: {normal_count.shape}")

    tflist = load_tflist({})
    log.info(f"TF list loaded: {len(tflist)} transcription factors")

    tfd, tgd = resolve_gene_lists(options, disease_count, tflist)
    tfn, tgn = resolve_gene_lists(options, normal_count,  tflist)

    log.info(f"Disease  — TFs: {len(tfd)}, targets: {len(tgd)}")
    log.info(f"Normal   — TFs: {len(tfn)}, targets: {len(tgn)}")

    # Intersect: only keep genes that passed filters in BOTH datasets
    common_targets = list(dict.fromkeys(set(tgd) & set(tgn)))
    common_tfs     = list(dict.fromkeys(set(tfd) & set(tfn)))

    log.info(f"Common targets (intersection): {len(common_targets)}")
    log.info(f"Common TFs    (intersection): {len(common_tfs)}")

    # Overlap sanity check: warn if sample names collide across datasets
    obs_overlap = set(disease_count.index) & set(normal_count.index)
    if obs_overlap:
        log.warning(f"Sample name overlap between datasets ({len(obs_overlap)} samples) — obs_names may not be unique after concat")
    else:
        log.info("Sample names are unique across datasets — OK")

    # Build AnnData objects — both use exactly common_targets so shapes match
    a_disease = ad.AnnData(X=disease_count[common_targets].values)
    a_disease.obs_names = disease_count.index.tolist()
    a_disease.var_names = common_targets
    a_disease.obs["condition"] = "disease"

    a_normal = ad.AnnData(X=normal_count[common_targets].values)
    a_normal.obs_names = normal_count.index.tolist()
    a_normal.var_names = common_targets
    a_normal.obs["condition"] = "normal"

    adata = ad.concat(
        [a_disease, a_normal],
        axis=0,
        join="inner",  # guardrail — both objects already aligned
        merge="same"
    ).copy()

    adata.uns["tf_genes"]     = common_tfs
    adata.uns["target_genes"] = common_targets

    log.info(f"Final AnnData shape: {adata.shape}  (samples × genes)")

    return adata


#  Plots 

def save_plots(adata: ad.AnnData, results_df: pd.DataFrame, out_dir: Path, log: logging.Logger):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Drop genes DESeq2 filtered out (NaN padj = too low counts to test)
    res = results_df.dropna(subset=["padj", "log2FoldChange"])
    sig = res[(res["padj"] < 0.05) & (res["log2FoldChange"].abs() > 1)]
    ns  = res[~res.index.isin(sig.index)]

    log.info(f"Plotting — significant genes (padj<0.05, |LFC|>1): {len(sig)}, non-significant: {len(ns)}")

    # 1. Volcano
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ns["log2FoldChange"],  -np.log10(ns["padj"]),  s=4, color="grey",    alpha=0.5, label="n.s.")
    ax.scatter(sig["log2FoldChange"], -np.log10(sig["padj"]), s=6, color="crimson",  alpha=0.7, label="sig (padj<0.05, |LFC|>1)")
    ax.axvline(x=1,  color="black", linestyle="--", linewidth=0.8)
    ax.axvline(x=-1, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(y=-np.log10(0.05), color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("log2 Fold Change (disease / normal)")
    ax.set_ylabel("-log10 adjusted p-value")
    ax.set_title("Volcano Plot")
    ax.legend(markerscale=2)
    fig.savefig(out_dir / "volcano.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. MA plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(np.log10(ns["baseMean"] + 1),  ns["log2FoldChange"],  s=4, color="grey",   alpha=0.5)
    ax.scatter(np.log10(sig["baseMean"] + 1), sig["log2FoldChange"], s=6, color="crimson", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("log10 mean expression (baseMean)")
    ax.set_ylabel("log2 Fold Change")
    ax.set_title("MA Plot")
    fig.savefig(out_dir / "ma_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. PCA — log1p for visualization only, NOT what DESeq2 used
    X_log  = np.log1p(adata.X)
    coords = PCA(n_components=2).fit(X_log)
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(X_log)
    conditions = adata.obs["condition"].values

    fig, ax = plt.subplots(figsize=(7, 6))
    for cond, color in [("disease", "crimson"), ("normal", "steelblue")]:
        mask = conditions == cond
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=cond, s=60, alpha=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("PCA — sample clustering")
    ax.legend()
    fig.savefig(out_dir / "pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Heatmap — top 50 most significant genes, z-scored per gene
    top_genes = res.sort_values("padj").head(50).index.tolist()
    top_genes = [g for g in top_genes if g in adata.var_names]

    expr    = adata[:, top_genes].X
    expr_df = pd.DataFrame(expr, index=adata.obs_names, columns=top_genes)
    z       = (expr_df - expr_df.mean()) / expr_df.std()
    z.index = adata.obs["condition"].values

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(z.T, cmap="RdBu_r", center=0, ax=ax,
                xticklabels=True, yticklabels=False,
                cbar_kws={"label": "z-score"})
    ax.set_title("Top 50 DE genes — expression heatmap")
    ax.set_xlabel("Samples")
    fig.savefig(out_dir / "heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Saved 4 plots to {out_dir}/")


#  Main entry point

def run(options: dict, output_path: str):
    out    = Path(output_path)
    log    = get_logger(out)
    rerun  = options.get("rerun", False)
    csv_path = out / "dge_results.csv"

    log.info("=" * 60)
    log.info(f"Run started — type: {options.get('type')}")
    log.info(f"Disease dataset : {options.get('disease_dataset')}")
    log.info(f"Normal dataset  : {options.get('normal_dataset')}")
    log.info(f"Rerun           : {rerun}")
    log.info(f"Options         : {options}")

    #  Rerun check 
    # If rerun=false and results.csv already exists, skip computation entirely
    if not rerun and csv_path.exists():
        log.info(f"results.csv found and rerun=false — loading cached results from {csv_path}")

        return pd.read_csv(csv_path, index_col=0),log

    if rerun and csv_path.exists():
        log.info("rerun=true — ignoring cached results.csv, recomputing")

    #  Run pipeline 
    adata = generate_input(options, log)

    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds  import DeseqStats

    log.info("Fitting DESeq2 model...")
    dds = DeseqDataSet(
        adata=adata,
        design_factors="condition",
        ref_level=["condition", "normal"]
    )
    dds.deseq2()
    log.info("DESeq2 model fitted")

    log.info("Running Wald test (disease vs normal)...")
    stat_res = DeseqStats(dds, contrast=["condition", "disease", "normal"])
    stat_res.summary()

    results_df = stat_res.results_df
    n_sig = results_df.dropna(subset=["padj"]).query("padj < 0.05").shape[0]
    log.info(f"DE genes (padj < 0.05): {n_sig} / {len(results_df)}")

    #  Save CSV
    results_df.sort_values("padj").to_csv(csv_path, index=False)
    log.info(f"Results saved to {csv_path}")

    #  Save plots 
    save_plots(adata, results_df, out/"dge_plots", log)

    log.info("Run complete")
    log.info("=" * 60)

    return results_df,log