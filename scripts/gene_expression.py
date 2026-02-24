import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from util import get_dataset
import json
import sys
from util_reports import save_plot_to_png, save_report_to_pdf, image_to_base64
import seaborn as sns
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BULK_EXPRESSION_DATASETS = [{"id": "bladder"}]

# stats methods


def get_tf_gene(ge, tf_list):
    """Extract TF genes present in gene expression matrix."""
    tf_found = [tf for tf in tf_list if tf in ge.columns]
    ge_tf = ge[tf_found]  # Only TFs that exist in ge
    return tf_found, ge_tf


def _compute_sparsity_distribution(ge, n_samples):
    """Internal helper: Compute sparsity distribution and stats for TF genes"""

    # Compute zero counts per gene
    # sum of zeros counted across rows for each column
    zero_counts = (ge == 0).sum(axis=0)

    # Create distribution DataFrame for each gene
    sparsity_df = pd.DataFrame({
        'zero_count': zero_counts.values,
        'gene_name': zero_counts.index
    })

    # Aggregate distribution
    distrib_df = sparsity_df.groupby('zero_count').size().reset_index()
    distrib_df.columns = ['zero_count', 'gene_count']

    # Sparsity stats
    stats = {
        'total_genes': len(zero_counts),
        'mean_sparsity': zero_counts.mean() / n_samples,
        'median_sparsity': zero_counts.median() / n_samples,
        'sparsest_gene': zero_counts.idxmax(),
        'sparsest_count': zero_counts.max(),
        'n_expressed_80': int((zero_counts / n_samples <= 0.2).sum())
    }
    return distrib_df, stats


def compute_stats(ge, tf_list, **args):
    """Main method: Basic stats + sparsity + extensible for future stats."""

    # total rows and col: ge(sample by genes)
    n_samples = ge.shape[0]
    n_genes = ge.shape[1]

    # ==== For transcription factors =======
    tf_found, ge_tf = get_tf_gene(ge, tf_list)
    sparsity_df_tf, sparsity_stats_tf = _compute_sparsity_distribution(
        ge_tf, n_samples)

    count_sum_tf = ge_tf.sum(axis=1)  # total sum of gene counts in each sample
    log_GE_tf = np.log2(ge_tf+1)

    # counts
    counts = compute_gene_rankings(ge_tf)
    count1 = duplicate_tf_cols(ge_tf)
    # ===== For target gene ==========
    target_genes = list(set(ge.columns)-set(tf_found))
    ge_target = ge[target_genes]
    sparsity_df_target, sparsity_stats_target = _compute_sparsity_distribution(
        ge_target, n_samples)
    log_GE_target = np.log2(ge_target+1)
    # total sum of gene counts in each sample
    count_sum_target = ge_target.sum(axis=1)
    # counts
    counts_target = compute_gene_rankings(ge_target)
    count1_target = duplicate_tf_cols(ge_target)

    return {
        'n_samples': n_samples,
        'n_genes': n_genes,

        'n_tf': len(tf_found),
        'genes_tf': tf_found,
        'sparsity_distribution_tf': sparsity_df_tf,
        'sparsity_stats_tf': sparsity_stats_tf,
        "gene_mean_tf": ge_tf.mean(axis=0),
        "gene_variance_tf": ge_tf.var(axis=0),
        "log_gene_mean_tf": log_GE_tf.mean(axis=0),
        "log_gene_variance_tf": log_GE_tf.var(axis=0),
        "log_GE_tf": log_GE_tf,
        "ge_tf": ge_tf,
        'sample_count_sum_tf': count_sum_tf,
        'counts_tf': {**counts, **count1},
        "n_expressed_80_tf": sparsity_stats_tf["n_expressed_80"],

        'n_target': len(target_genes),
        'genes_target': target_genes,
        'sparsity_distribution_target': sparsity_df_target,
        'sparsity_stats_target': sparsity_stats_target,
        "gene_mean_target": ge_target.mean(axis=0),
        "gene_variance_target": ge_target.var(axis=0),
        "log_gene_mean_target": log_GE_target.mean(axis=0),
        "log_gene_variance_target": log_GE_target.var(axis=0),
        "log_GE_target": log_GE_target,
        "ge_target": ge_target,
        'sample_count_sum_target': count_sum_target,
        'counts_target': {**counts_target, **count1_target},
        "n_expressed_80_target": sparsity_stats_target["n_expressed_80"],
    }


def compute_gene_rankings(ge_tf, top_n=20):
    stats = pd.DataFrame({
        'mean': ge_tf.mean(axis=0),
        'std': ge_tf.std(axis=0),
        'min': ge_tf.min(axis=0),
        'max': ge_tf.max(axis=0)
    })
    stats = stats.sort_values('mean', ascending=False)
    
    # ADD .reset_index() to make gene name a column instead of index
    top_high = stats.head(top_n).reset_index()
    top_high.columns = ['gene', 'mean', 'std', 'min', 'max']  # Rename columns
    
    top_low = stats.tail(top_n).reset_index()
    top_low.columns = ['gene', 'mean', 'std', 'min', 'max']

    return {"top_high": top_high, "top_low": top_low}


def duplicate_tf_cols(df):
    """
    Return a list of duplicate gene cols found in the data
    """
    col_counts = pd.Series(df.columns.tolist()).value_counts()
    duplicates = col_counts[col_counts > 1]

    if duplicates.empty:
        print("No duplicate columns found.")
        return pd.DataFrame()

    result = pd.DataFrame({
        'name': duplicates.index,
        'count': duplicates.values
    })

    return {"duplicate": result.sort_values('count', ascending=False)}

# plot methods


def add_info_box(ax, content, **kwargs):
    """Add info box at bottom-right corner without overlapping plot."""
    # Default styling
    box_props = {
        'facecolor': 'white',
        'edgecolor': 'gray',
        'boxstyle': 'round,pad=0.3',
        'alpha': 0.9,
        'fontsize': 8
    }
    box_props.update(kwargs)

    # Create text with timestamp
    timestamp = datetime.now().strftime("Created: %Y-%m-%d %H:%M")
    full_text = f"{content}\n{timestamp}"

    # Add text box at bottom-right (anchored position)
    ax.text(0.98, 0.02, full_text, transform=ax.transAxes, fontsize=box_props['fontsize'],
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle=box_props['boxstyle'],
                      facecolor=box_props['facecolor'],
                      edgecolor=box_props['edgecolor'],
                      alpha=box_props['alpha']))


def plot_sparsity_distribution(stats_dict, gene_type="tf", title="Sparsity distribution", **args):
    """Plot binned sparsity distribution with threshold lines T=0.10, T=0.20, etc."""
    # 'sparsity_distribution_tf'
    distrib_df = stats_dict[f"sparsity_distribution_{gene_type}"]
    sparsity_stats = stats_dict[f"sparsity_stats_{gene_type}"]
    n_samples = stats_dict['n_samples']

    # Convert to non-zero counts
    distrib_df['non_zero_count'] = n_samples - distrib_df['zero_count']

    # Adaptive bin size based on number of samples
    if n_samples <= 50:
        bin_size = 1
    elif n_samples <= 200:
        bin_size = 5
    elif n_samples <= 500:
        bin_size = 10
    else:
        bin_size = max(20, n_samples // 25)

    # Create bins
    bins = np.arange(0, n_samples + bin_size, bin_size)
    distrib_df['non_zero_bin'] = pd.cut(
        distrib_df['non_zero_count'], bins, right=False)

    # Aggregate binned counts
    binned_df = distrib_df.groupby('non_zero_bin', observed=False)[
        'gene_count'].sum().reset_index()
    binned_df['non_zero_bin_mid'] = [
        interval.mid for interval in binned_df['non_zero_bin']]

    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Plot binned bars
    bars = plt.bar(binned_df['non_zero_bin_mid'], binned_df['gene_count'],
                   width=bin_size*0.9, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add count labels on top of bars
    for bar, count in zip(bars, binned_df['gene_count']):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + max(0.5, height*0.02),
                    f'{int(count)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Mean non-zero line
    mean_nonzero = n_samples * (1 - sparsity_stats['mean_sparsity'])
    plt.axvline(mean_nonzero, color='red', linestyle='--', linewidth=2,
                label=f'Mean non-zeros: {mean_nonzero:.1f}')

    # THRESHOLD LINES: T=0.10, T=0.20, etc. (minimum non-zero fraction)
    threshold_fractions = [0.05, 0.10, 0.15, 0.20, 0.25]  # T values
    colors = ['green', 'blue', 'orange', 'purple', 'brown']

    for T, color in zip(threshold_fractions, colors):
        threshold_nonzero = T * n_samples  # Minimum non-zeros required
        genes_retained = distrib_df[distrib_df['non_zero_count']
                                    >= threshold_nonzero]['gene_count'].sum()

        plt.axvline(threshold_nonzero, color=color, linestyle=':', linewidth=2, alpha=0.8,
                    label=f'T={T:.0%} {genes_retained} TFs')

    plt.xlabel(f'Non-zero count per {gene_type} gene (binned)')
    plt.ylabel('Number of genes')
    plt.title(f'{gene_type} gene expression sparsity distribution')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Info box
    count = stats_dict[f"n_{gene_type}"]
    info_content = (f"n_samples: {n_samples}\nn_gene: {count}")
    add_info_box(ax, info_content)

    plt.ylim(0, max(binned_df['gene_count']) * 1.15)
    plt.tight_layout()
    return plt


def plot_gene_coverage(stats, tau: float = 0, thresholds=[0.5, 0.8, 0.85, 0.9, 0.95], gene_type="tf"):
    """
    Plot F(t) — fraction of genes retained as threshold t increases.

    Args:
        GE:         samples by genes DataFrame
        tau:        expression threshold for computing coverage (i.e the value that is considered zero , can be 0 but also 1 )
        thresholds: list of t values to highlight
    """
    GE = stats[f"ge_{gene_type}"]
    coverage = (GE > tau).mean(axis=0)

    t_values = np.linspace(0, 1, 200)
    f_values = [(coverage >= t).mean() for t in t_values]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_values, f_values, color='steelblue', linewidth=1)

    for t in thresholds:
        f_t = (coverage >= t).mean()
        n_genes = int(f_t * len(coverage))
        ax.axvline(t, linestyle='--',
                   label=f't={t:.0%} → {n_genes} genes ({f_t:.0%})', linewidth=1)
        ax.axhline(f_t, linestyle=':', alpha=0.4, linewidth=1)

    ax.set_xlabel('Threshold t')
    ax.set_ylabel('F(t) — fraction of genes retained')
    ax.set_title(
        f' {gene_type} gene coverage \nn={GE.shape[0]} samples, m={GE.shape[1]} genes')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def plot_sample_count_sum(stats_dict, gene_type="tf", **args):
    """Plot binned sparsity distribution with threshold lines T=0.10, T=0.20, etc."""
    sample_sums = stats_dict[f'sample_count_sum_{gene_type}']

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    plt.bar(range(len(sample_sums)), sample_sums.sort_values(),
            alpha=0.7, edgecolor='black', linewidth=0.5)

    plt.xlabel('Sample index')
    plt.ylabel(f'Sum of {gene_type} gene counts')
    plt.title(f'Sum of genes counts per sample')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    # add_info_box(ax, info_content)
    plt.tight_layout()
    return plt


def plot_mean_variance_comparison(stats_dict, gene_type="tf", meta=None, **args):
    """
    Plot side-by-side mean-variance for raw and log-transformed data.
    Shows 10th and 20th percentile thresholds for filtering low-expression genes.

    Args:
        stats_dict: Dictionary with gene_mean, gene_variance, log_gene_mean, log_gene_variance
        meta: Optional metadata dict
    """
    raw_mean = stats_dict[f'gene_mean_{gene_type}']
    raw_var = stats_dict[f'gene_variance_{gene_type}']
    log_mean = stats_dict[f'log_gene_mean_{gene_type}']
    log_var = stats_dict[f'log_gene_variance_{gene_type}']

    # Calculate 10th and 20th percentile thresholds
    raw_p10 = raw_mean.quantile(0.10)
    raw_p20 = raw_mean.quantile(0.20)
    log_p10 = log_mean.quantile(0.10)
    log_p20 = log_mean.quantile(0.20)

    # Count genes that would be KEPT (above threshold)
    raw_n10 = (raw_mean >= raw_p10).sum()
    raw_n20 = (raw_mean >= raw_p20).sum()
    log_n10 = (log_mean >= log_p10).sum()
    log_n20 = (log_mean >= log_p20).sum()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # LEFT: Raw Data
    ax1.scatter(raw_mean, raw_var, s=2, alpha=0.5, color='steelblue')
    # this is for better visualization
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.axvline(raw_p10, color='red', linestyle='--', linewidth=2,
                label=f'10th percentile (≥{raw_p10:.2f}): {raw_n10} genes kept')
    ax1.axvline(raw_p20, color='orange', linestyle='--', linewidth=2,
                label=f'20th percentile (≥{raw_p20:.2f}): {raw_n20} genes kept')

    ax1.set_xlabel('Mean Expression (log scale)', fontsize=12)
    ax1.set_ylabel('Variance (log scale)', fontsize=12)
    ax1.set_title('Raw TPM Data - Filter Thresholds',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3, which='both')

    # RIGHT: Log Data
    ax2.scatter(log_mean, log_var, s=2, alpha=0.5, color='darkgreen')

    ax2.axvline(log_p10, color='red', linestyle='--', linewidth=2,
                label=f'10th percentile (≥{log_p10:.2f}): {log_n10} genes kept')
    ax2.axvline(log_p20, color='orange', linestyle='--', linewidth=2,
                label=f'20th percentile (≥{log_p20:.2f}): {log_n20} genes kept')

    ax2.set_xlabel(f'Mean Expression for {gene_type}', fontsize=12)
    ax2.set_ylabel('Variance', fontsize=12)
    ax2.set_title('Log2(TPM+1) Data - Filter Thresholds',
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    return plt


def plot_pca(stats_dict, meta=None, **args):
    # Log transform first: log2(TPM+1)
    log_ge = stats_dict["log_GE"]  # np.log2(ge_df + 1)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(StandardScaler().fit_transform(log_ge))
    plt.scatter(coords[:, 0], coords[:, 1])
    plt.title("PCA of Samples")
    return plt


# cli methods

def genes_to_compact_text(genes, max_per_line=10):
    """
    Display genes as comma-separated text instead of table (even more compact).
    
    Args:
        genes: List of gene names
        max_per_line: Maximum genes per line before wrapping
    
    Returns:
        HTML string with genes as comma-separated text
    """
    if not genes:
        return "<p><em>No duplicate genes found</em></p>"
    
    # Split into chunks for line breaks
    chunks = [genes[i:i+max_per_line] for i in range(0, len(genes), max_per_line)]
    lines = [", ".join(chunk) for chunk in chunks]
    
    return '<p style="font-size: 6pt; line-height: 1.4;">' + "<br>".join(lines) + "</p>"

def genes_to_table(genes, cols=8):  # Changed from 6 to 8 columns for more compact layout
    """
    Create a compact HTML table from a list of gene names.
    
    Args:
        genes: List of gene names
        cols: Number of columns (default 8 for more compact layout)
    
    Returns:
        HTML string for a compact table
    """
    if not genes:
        return "<p><em>No duplicate genes found</em></p>"
    
    remainder = len(genes) % cols
    if remainder:
        genes = genes + [""] * (cols - remainder)
    
    rows = [genes[i:i+cols] for i in range(0, len(genes), cols)]
    
    # Create compact table with minimal styling
    rows_html = "".join(
        "<tr>" + "".join(f"<td>{g if g else '&nbsp;'}</td>" for g in row) + "</tr>"
        for row in rows
    )
    
    return f'<table style="font-size: 5pt; margin: 4pt 0;"><tbody>{rows_html}</tbody></table>'

def generate_GE_report(id, stats, plot_paths, output_path, meta, replace):
    from util_reports import image_to_base64  # Add this import
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    duplicate_genes_tf = stats["counts_tf"]["duplicate"]["name"].tolist()
    duplicate_genes_target = stats["counts_target"]["duplicate"]["name"].tolist()


    html_content = f"""
        <h1>Gene Expression Report</h1>

        <p>Source: {meta.get('title', 'Dataset')}<br>
        Downloaded: {meta.get('downloaded', '')}<br>
        Generated: {timestamp}</p>

        <h2>Basic Stats</h2>
        <p>Samples: {stats["n_samples"]} &nbsp;|&nbsp;
        Genes: {stats["n_genes"]} &nbsp;|&nbsp;
        TF: {stats["n_tf"]} &nbsp;|&nbsp;
        Targets: {stats["n_target"]}</p>

        <hr>

        <h2>TF Stats</h2>

        <h3>Sparsity</h3>
        
        <img src="{image_to_base64(plot_paths['sparsity_tf0'])}" style="max-width: 100%;">
        <p> Coverage of gene as a function of coverage threshold. Coverage is expression value greater than 0  </p>
        
        <img src="{image_to_base64(plot_paths['sparsity_tf1'])}" style="max-width: 100%;">
        <p> Coverage of gene as a function of coverage threshold. Coverage is expression value greater than 1  </p>

        <img src="{image_to_base64(plot_paths['sparsity_tf'])}" style="max-width: 100%;">
        <p>Distribution of non-zero values across TFs. X-axis shows sample count.</p>


        <h3>Mean Variance</h3>
        <img src="{image_to_base64(plot_paths['mean_variance_tf'])}" style="max-width: 100%;">

        <h3>Highly Expressed</h3>
        {stats["counts_tf"]["top_high"].to_html(index=False)}

        <h3>Lowly Expressed</h3>
        {stats["counts_tf"]["top_low"].to_html(index=False)}

        <h3>Duplicates</h3>
        {genes_to_table(duplicate_genes_tf)}

        <hr>

        <h2>Target Gene Stats</h2>

        <h3>Sparsity</h3>
        <img src="{image_to_base64(plot_paths['sparsity_target0'])}" style="max-width: 100%;">
        <p> Coverage of gene as a function of coverage threshold. Coverage is expression value greater than 0  </p>
        
        <img src="{image_to_base64(plot_paths['sparsity_target1'])}" style="max-width: 100%;">
        <p> Coverage of gene as a function of coverage threshold. Coverage is expression value greater than 1  </p>

        <img src="{image_to_base64(plot_paths['sparsity_target'])}" style="max-width: 100%;">
        <p>Distribution of non-zero values across Target genes. X-axis shows sample count.</p>
        
        <h3>Mean Variance</h3>
        <img src="{image_to_base64(plot_paths['mean_variance_target'])}" style="max-width: 100%;">

        <h3>Highly Expressed</h3>
        {stats["counts_target"]["top_high"].to_html(index=False)}

        <h3>Lowly Expressed</h3>
        {stats["counts_target"]["top_low"].to_html(index=False)}

        <h3>Duplicates</h3>
        {genes_to_compact_text(duplicate_genes_target)}
    """

    save_report_to_pdf(html_content, f"{id}_stats", output_path, replace)



def generate_plots(stats, output_dir, temp_dir, expr_meta=None, replace=True, expr_df=None):
    """
    Generate all plots for a single gene expression file.
    If replace=False and plots exist, skip generation but return existing paths.
    """
    import os
    from pathlib import Path
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Helper to check if plot exists and handle accordingly
    def get_or_create_plot(plot_func, plot_name, *args, **kwargs):
        plot_path = Path(temp_dir) / f"{plot_name}.png"
        
        if plot_path.exists() and not replace:
            print(f"Using existing plot: {plot_path}")
            return plot_path
        else:
            # Generate new plot
            plt_obj = plot_func(*args, **kwargs)
            return save_plot_to_png(plt_obj, plot_name, temp_dir, replace)
    
    return {
        'sparsity_tf': get_or_create_plot(plot_sparsity_distribution, "sparsity_tf", stats, gene_type="tf"),
        'sparsity_tf0': get_or_create_plot(plot_gene_coverage, "sparsity_tf0", stats, gene_type="tf", tau=0),
        'sparsity_tf1': get_or_create_plot(plot_gene_coverage, "sparsity_tf1", stats, gene_type="tf", tau=1),

        'sparsity_target': get_or_create_plot(plot_sparsity_distribution, "sparsity_target", stats, gene_type="target"),
        'sparsity_target0': get_or_create_plot(plot_gene_coverage, "sparsity_target0", stats, gene_type="target", tau=0),
        'sparsity_target1': get_or_create_plot(plot_gene_coverage, "sparsity_target1", stats, gene_type="target", tau=1),

        'mean_variance_tf': get_or_create_plot(plot_mean_variance_comparison, "mean_variance_tf", stats, gene_type="tf"),
        'mean_variance_target': get_or_create_plot(plot_mean_variance_comparison, "mean_variance_target", stats, gene_type="target")
    }

def Generate_Reports(dataset_list, rerun_plots=False,rerun_summary=False, output_dir=None):

    _, tf_data = get_dataset("tflist")
    tf_names = tf_data['gene_name'].tolist()

    dataset_loc = Path(os.getenv("DATA_PATH"))
    summary_list = []

    for ds in dataset_list:
        ge = ds["id"]

        # if output_dir is None:
        #    output_dir = dataset_loc/f"{ge}"
        ds_output_dir = output_dir if output_dir is not None else dataset_loc / \
            f"{ge}"
        temp_folder = ds_output_dir / "temp" / f"{ge}"
        file_check = ds_output_dir / f"{ge}_stats.pdf"
        gen_plots = True
        if file_check.exists() and not rerun_plots:
            print(f' {ge} file exists and rerun not set. skipping')
            gen_plots = False
        
        if not gen_plots and not rerun_summary :
            continue

        
        print(f"Generating report for {ge}...")
        # print(meta)
        meta, data = get_dataset(ge)
        stats = compute_stats(data, tf_names)
        if gen_plots:
            plot_paths = generate_plots(
                stats=stats,
                expr_df=data,
                output_dir=ds_output_dir,
                temp_dir=temp_folder,
                expr_meta=meta,
                replace=rerun_plots)
            generate_GE_report(ge, stats, plot_paths, ds_output_dir, meta, rerun_plots)

        summary_list.append(
            {   
                "id":ds["id"],
                "title": meta["title"],
                "n_samples": stats["n_samples"],
                "n_total_genes": stats["n_genes"],


                "n_tf": stats["n_tf"],
                "n_expressed_80_tf": stats["n_expressed_80_tf"],
                "mean_gene_mean_tf": round(float(stats["gene_mean_tf"].mean()), 4),
                "mean_gene_variance_tf": round(float(stats["gene_variance_tf"].mean()), 4),
                "n_duplicate_tf": len(stats["counts_tf"]["duplicate"]) if not stats["counts_tf"]["duplicate"].empty else 0,     
                
                
                "n_targets": stats["n_target"],
                "n_expressed_80_target": stats["n_expressed_80_target"],
                "mean_gene_mean_target": round(float(stats["gene_mean_target"].mean()), 4),
                "n_duplicate_target": len(stats["counts_target"]["duplicate"]) if not stats["counts_target"]["duplicate"].empty else 0,
                
            }
        )
        print(f"Report saved: {ds_output_dir}")
    
    summary_df = pd.DataFrame(summary_list)
    summary_path = Path(output_dir) / "datasets_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")
    
    print("done!")

if __name__ == "__main__":
    # Accept first command line argument as module name
    # load_env()
    module_arg = sys.argv[1] if len(sys.argv) > 2 else "report"
    rerun_arg = sys.argv[2] if len(sys.argv) > 2 else "0"
    rerun = rerun_arg.lower() in ["1", "t", "true", "y"]
    # print(module_arg,rerun)
    if module_arg == "report":
        Generate_Reports(dataset_list=[{"id": "amygdala"}], rerun=rerun)
    else:
        print(f"Invalid command: {module_arg}")
