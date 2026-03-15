# FEA.py : functional_enrichment
from gprofiler import GProfiler
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np


def select_gene_lists(
    results_df: pd.DataFrame,
    padj_thresh: float = 0.05,
    lfc_thresh: float = 1.0,
    top_n: int = 500,
    log: logging.Logger = None,
) -> dict[str, list[str]]:
    """
    Extract up/down-regulated gene lists from DESeq2 results.
    Falls back to top_n ranking by padj if the filtered list is very large.
    """
    res = results_df.dropna(subset=["padj", "log2FoldChange"])

    up = res[(res["padj"] < padj_thresh) & (
        res["log2FoldChange"] > lfc_thresh)]
    down = res[(res["padj"] < padj_thresh) & (
        res["log2FoldChange"] < -lfc_thresh)]

    # Sort by padj, then cap at top_n
    up = up.sort_values("padj").head(top_n)
    down = down.sort_values("padj").head(top_n)

    gene_lists = {
        "upregulated":   up.index.tolist(),
        "downregulated": down.index.tolist(),
    }

    if log:
        log.info(
            f"Gene lists for enrichment — up: {len(gene_lists['upregulated'])}, down: {len(gene_lists['downregulated'])}")

    return gene_lists


def run_gprofiler(
    gene_lists: dict[str, list[str]],
    organism: str = "hsapiens",
    sources: list[str] = None,
    log: logging.Logger = None,
) -> pd.DataFrame:
    """
    Run g:Profiler functional enrichment on multiple gene lists.
    Returns a combined DataFrame with a 'query' column identifying the list.
    """
    if sources is None:
        sources = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC", "TF", "WP"]

    # Filter out empty lists before sending
    non_empty = {k: v for k, v in gene_lists.items() if v}
    if not non_empty:
        if log:
            log.warning("All gene lists are empty — skipping g:Profiler")
        return pd.DataFrame()

    gp = GProfiler(return_dataframe=True)

    results = gp.profile(
        organism=organism,
        query=non_empty,          # dict → multi-query mode
        sources=sources,
        significance_threshold_method="fdr",
        user_threshold=0.05,
        no_evidences=False,       # keep intersection genes for downstream use
    )

    if log:
        log.info(
            f"g:Profiler returned {len(results)} significant terms across {len(non_empty)} queries")

    return results


def save_enrichment_plots(
    enrichment_df: pd.DataFrame,
    out_dir: Path,
    top_n_terms: int = 20,
    log: logging.Logger = None,
):
    """
    Save bar charts of top enriched terms per query (up/down separately).
    """
    if enrichment_df.empty:
        if log:
            log.warning("No enrichment results — skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    queries = enrichment_df["query"].unique()

    for query in queries:
        df = (
            enrichment_df[enrichment_df["query"] == query]
            .sort_values("p_value")
            .head(top_n_terms)
            .copy()
        )

        if df.empty:
            continue

        df["-log10(p)"] = -np.log10(df["p_value"].clip(lower=1e-300))
        df = df.sort_values("-log10(p)")

        fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)))
        colors = [
            "#c0392b" if src.startswith("GO") else
            "#2980b9" if src == "KEGG" else
            "#27ae60" if src == "REAC" else
            "#8e44ad"
            for src in df["source"]
        ]
        ax.barh(df["name"], df["-log10(p)"], color=colors)
        ax.set_xlabel("-log10(adjusted p-value)")
        ax.set_title(f"Top {top_n_terms} enriched terms — {query}")
        ax.axvline(x=-np.log10(0.05), color="black",
                   linestyle="--", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"enrichment_{query}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        if log:
            log.info(f"Saved enrichment plot for '{query}' ({len(df)} terms)")

    # Dot plot — top terms across all queries by source
    _save_dotplot(enrichment_df, out_dir, top_n_terms, log)


def _save_dotplot(
    enrichment_df: pd.DataFrame,
    out_dir: Path,
    top_n_terms: int,
    log: logging.Logger,
):
    """Dot plot: term vs query, dot size = gene ratio, color = -log10(p)."""
    df = enrichment_df.copy()
    df["gene_ratio"] = df["intersection_size"] / df["query_size"]
    df["-log10(p)"] = -np.log10(df["p_value"].clip(lower=1e-300))

    # Pick top N terms by best p-value across any query
    top_terms = (
        df.groupby("name")["-log10(p)"].max()
        .sort_values(ascending=False)
        .head(top_n_terms)
        .index.tolist()
    )
    df = df[df["name"].isin(top_terms)]

    pivot_p = df.pivot_table(
        index="name", columns="query", values="-log10(p)",  fill_value=0)
    pivot_ratio = df.pivot_table(
        index="name", columns="query", values="gene_ratio", fill_value=0)

    queries = pivot_p.columns.tolist()
    terms = pivot_p.index.tolist()

    fig, ax = plt.subplots(
        figsize=(4 + len(queries) * 1.5, max(6, len(terms) * 0.4)))

    for i, term in enumerate(terms):
        for j, query in enumerate(queries):
            p_val = pivot_p.loc[term, query]
            ratio = pivot_ratio.loc[term, query]
            if p_val > 0:
                ax.scatter(j, i, s=ratio * 800, c=p_val,
                           cmap="YlOrRd", vmin=0, vmax=pivot_p.values.max(),
                           edgecolors="grey", linewidths=0.4)

    ax.set_xticks(range(len(queries)))
    ax.set_xticklabels(queries, rotation=30, ha="right")
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms, fontsize=8)
    ax.set_title("Enrichment dot plot (size = gene ratio, color = -log10 p)")

    sm = plt.cm.ScalarMappable(
        cmap="YlOrRd", norm=plt.Normalize(0, pivot_p.values.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="-log10(adjusted p-value)")
    fig.tight_layout()
    fig.savefig(out_dir / "enrichment_dotplot.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    if log:
        log.info(
            f"Saved enrichment dot plot ({len(terms)} terms × {len(queries)} queries)")


def run(
    results_df: pd.DataFrame,
    out_dir: Path,
    options: dict,
    log: logging.Logger,
) -> pd.DataFrame:
    """
    Entry point — mirrors DGE.run() signature style.
    Reads enrichment options from the same options dict.
    """
    csv_path = out_dir / "enrichment_results.csv"
    rerun = options.get("rerun", False)

    if not rerun and csv_path.exists():
        log.info(
            f"enrichment_results.csv found and rerun=false — loading cached results")
        return pd.read_csv(csv_path)

    gene_lists = select_gene_lists(
        results_df,
        padj_thresh=options.get("padj_thresh", 0.05),
        lfc_thresh=options.get("lfc_thresh", 1.0),
        top_n=options.get("enrichment_top_n", 500),
        log=log,
    )

    enrichment_df = run_gprofiler(
        gene_lists,
        organism=options.get("organism", "hsapiens"),
        sources=options.get("enrichment_sources",  ["GO:BP", "KEGG", "REAC"]),
        log=log,
    )

    if not enrichment_df.empty:
        enrichment_df["intersections"] = enrichment_df["intersections"].apply(lambda x: ";".join(x))
        enrichment_df["parents"]       = enrichment_df["parents"].apply(lambda x: ";".join(x))
        enrichment_df["evidences"]     = enrichment_df["evidences"].apply(lambda x: ";".join(["|".join(e) for e in x]))
        enrichment_df.to_csv(csv_path, index=False)
        log.info(f"Enrichment results saved to {csv_path}")
        save_enrichment_plots(enrichment_df, out_dir /
                              "enrichment_plots", log=log)
    else:
        log.warning("No significant enrichment terms found")

    return enrichment_df
