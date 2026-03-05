#!/usr/bin/env python3
"""
To generate result
"""

import json
import os
import sys
import sqlite3
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
from io import BytesIO

from tfitpy import compute_indices

from util import TOOLS, get_exp_path, get_temp_path, get_output_path, get_data_path, get_dataset


INDICES_LIST = ["goa_similarity","shortest_PPI_path_score_hippie","shortest_PPI_path_score_stringdb","shortest_PPI_path_score_biogrid","shared_PPI_partners_score_hippie","shared_PPI_partners_score_stringdb","shared_PPI_partners_score_biogrid","grn_collectri"]

INDICES = {
    "goa_similarity_lin": {
        "title": "GO Similarity (Lin)",
        "plot":"box"
    },
    "goa_similarity_resnik": {
        "title": "GO Similarity (Resnik)",
        "plot":"box"
    },
    "goa_similarity_jc": {
        "title": "GO Similarity (JC)",
        "plot":"box"
    },
    "shortest_PPI_path_score_hippie": {
        "title": "PPI Shortest Path (Hippie)",
        "plot":"box"
    },
     "shortest_PPI_path_score_stringdb": {
        "title": "PPI Shortest Path (StringDB)",
        "plot":"box"
    },
      "shortest_PPI_path_score_biogrid": {
        "title": "PPI Shortest Path (Biogrid)",
        "plot":"box"
    },
      "shared_PPI_partners_score_hippie": {
        "title": "PPI Source Partners (Hippie)",
        "plot":"box"
    },
      "shared_PPI_partners_score_stringdb": {
        "title": "PPI Source Partners (StringDB)",
        "plot":"box"
    },
    "shared_PPI_partners_score_biogrid": {
        "title":"PPI Source Partners (Biogrid)",
        "plot":"box"
    },
    "grn_collectri_precision": {
        "title":"GRN Precision (Collectri)",
        "plot":"box"
    },
    "grn_collectri_recall": {
        "title":"GRN Recall (Collectri)",
        "plot":"box"
    },
    # "grn_precision_recall_collectri": {
    #     "title": "D",
    #     "plot":"line"
    # },
    # "grn_set_metrics_collectri": {
    #     "title": "E",
    #     "plot":"none"
    # },
}

# ------------------------------------------
# Publication style constants
# ------------------------------------------
FONT_SIZE_LABEL   = 10    # axis labels
FONT_SIZE_TICK    = 9     # tick labels
FONT_SIZE_PANEL   = 14    # panel letter  (A), (B) …
FONT_Y_LABEL      = 12
# Seaborn palette – muted, grey-friendly, print-safe
_PALETTE = ["#058ED9", "#f4ebd9", "#483d3f", "#a39a92", "#77685d"]


def _apply_base_style(ax):
    """Strip chart junk; minimal publication look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK,
                   length=2, width=0.5, pad=1)
    ax.grid(axis="y", color="0.85", linewidth=0.4, linestyle="--", zorder=0)
    ax.set_axisbelow(True)


# ----------------------
# Data preparation
# ----------------------

def prepare_combined_dataframe(tools_data):
    """Prepare combined dataframe with common targets."""
    all_targets = [set(df['target'].unique()) for df, _ in tools_data.values()]
    common_targets = set.intersection(*all_targets)

    final_dfs = []
    for tool_name, (df, _) in tools_data.items():
        df_filtered = df[df['target'].isin(common_targets)].copy()
        df_filtered['plot_tool'] = tool_name
        final_dfs.append(df_filtered)

    final_df = pd.concat(final_dfs, ignore_index=True)
    return final_df, common_targets



def compute_score(data):
    """Compute scores for the given data."""
    if data is None:
        raise ValueError("No data provided")
    mask = data['sources'].str.split(";").str.len() > 1
    data = data[mask].copy()
    bio_data_path = Path(os.path.expandvars(os.getenv("DATA_PATH")))
    df_score, add_data = compute_indices(df=data,  data_path=bio_data_path)
    return df_score, add_data


def get_indices(exp_name, dataset, tool, result_file_name,rerun=False):
    """Generate indices for tool results if not already generated and return the values."""
    result_file       = get_output_path() / f"{exp_name}/{dataset}/{tool}/{result_file_name}.csv"
    #indices_file_csv  = get_output_path() / f"{exp_name}/{dataset}/{tool}/{result_file_name}_score.csv"
    indices_file_json = get_output_path() / f"{exp_name}/{dataset}/{tool}/{result_file_name}_score.json"

    score_file = pd.read_csv(result_file)
    # with open(indices_file_json, "r") as f:
    #     score_file_json = json.load(f)

    score_file, score_file_json = compute_score(score_file)
    score_file.to_csv(result_file, index=False)
    # with open(indices_file_json, "w") as f:
    #     json.dump(score_file_json, f, indent=2)
    return score_file, score_file_json


def combine_dataset_tool_data(all_dataset_tool_data):
    """
    Merge tool data across multiple datasets into a single tools_data dict.

    Parameters
    ----------
    all_dataset_tool_data : list of dicts
        Each element is a tool_data dict {tool_name: (df, meta)} from one dataset.

    Returns
    -------
    combined_tools_data : dict  {tool_name: (combined_df, meta)}
    """
    combined_tools_data = {}
    for tool_name in TOOLS.keys():
        dfs = []
        meta = {}
        for tool_data in all_dataset_tool_data:
            if tool_name in tool_data:
                df, meta = tool_data[tool_name]
                dfs.append(df)
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_tools_data[tool_name] = (combined_df, meta)
    return combined_tools_data

# -----------------------
# Single-panel drawing
# -----------------------

def create_index_subplot(ax, final_df, tools_data, index_name, plot_type, panel_label, y_label=None):
    """
    Draw one subplot panel using seaborn for publication-quality output.

    Returns
    -------
    plot_df : DataFrame used for this subplot (columns: plot_tool, target, sources, method, <index_name>),
              or None if no data / not applicable.
    """
    tool_keys = list(tools_data.keys())
    tool_order = [TOOLS[key]["title"] for key in tool_keys]

    if plot_type == "box":
        rows = []
        for tool_key in tool_keys:
            sub = final_df[final_df["plot_tool"] == tool_key]
            if sub.empty or index_name not in sub.columns:
                continue
            best = sub.loc[sub.groupby("target")[index_name].idxmax()]
            rows.append(best[["plot_tool", "target", "sources", "method", index_name]])

        if not rows:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    fontsize=FONT_SIZE_TICK, color="0.5", transform=ax.transAxes)
            ax.set_visible(True)
            return None

        plot_df = pd.concat(rows, ignore_index=True)
        plot_df["plot_tool"] = plot_df["plot_tool"].map(lambda x: TOOLS[x]["title"])

        sns.boxplot(
            data=plot_df,
            x="plot_tool",
            legend=False,
            hue="plot_tool",
            y=index_name,
            order=tool_order,
            palette=_PALETTE[: len(tool_order)],
            width=0.50,
            linewidth=0.6,
            flierprops=dict(
                marker="o",
                markerfacecolor="0.4",
                markeredgewidth=0,
                markersize=1.5,
                alpha=0.5,
            ),
            whiskerprops=dict(linewidth=0.7),
            capprops=dict(linewidth=0.7),
            ax=ax,
        )

        ax.set_xlabel("")
        #print(y_label)
        ax.set_ylabel(y_label , fontsize=FONT_Y_LABEL,
                      labelpad=1, fontweight="normal")
        ax.tick_params(axis="x", labelsize=FONT_SIZE_TICK)
        for lbl in ax.get_xticklabels():
            t = lbl.get_text()
            if len(t) > 12:
                lbl.set_text(t[:12])
            lbl.set_ha("center")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune="both"))

        _apply_base_style(ax)
        ax.text(-0.18, 1.06, f"({panel_label})", transform=ax.transAxes,
                fontsize=FONT_SIZE_PANEL, fontweight="bold", va="top", ha="left",
                clip_on=False)
        return plot_df




# ------------------------
# Multi-panel figure
# ------------------------

def create_index_plots(tools_data, indices, width_cm=7, height_cm=9.9,
                       n_cols_max=4, dpi=300):
    """
    Create a publication-ready main image with subplots for each index.

    Parameters
    ----------
    tools_data : dict  {tool_name: (dataframe, meta_dict)}
    indices    : dict  {index_name: {'title': str, 'plot': plot_type}}
    width_cm   : float  figure width in cm
    height_cm  : float  figure height in cm
    n_cols_max : int    maximum number of columns
    dpi        : int    resolution for raster export

    Returns
    -------
    fig        : matplotlib Figure
    index_dfs  : dict  {index_name: DataFrame}  — data used per subplot
    """
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         FONT_SIZE_LABEL,
        "axes.linewidth":    0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size":  2,
        "ytick.major.size":  2,
        "pdf.fonttype":      42,   # editable text in Illustrator/Inkscape
        "ps.fonttype":       42,
    })

    final_df, common_targets = prepare_combined_dataframe(tools_data)
    # print(f"Common targets: {len(common_targets)}")

    plottable = {k: v for k, v in indices.items() if v.get("plot") != "none"}
    n_plots = len(plottable)
    if n_plots == 0:
        print("No plottable indices.")
        return None, {}

    n_cols = min(n_cols_max, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    print(f"Layout: {n_rows} rows x {n_cols} cols")

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(width_cm, height_cm),
                             squeeze=False)
    axes_flat = axes.flatten()

    index_dfs = {}
    for idx, (index_name, cfg) in enumerate(plottable.items()):
        panel_label = chr(65 + idx)   # A, B, C, …
        ax = axes_flat[idx]
        print(f"  ({panel_label}) {cfg['plot']}  -  {index_name}")
        plot_df = create_index_subplot(ax, final_df, tools_data,
                                       index_name, cfg["plot"], panel_label,
                                       y_label=cfg.get("title"))
        if plot_df is not None:
            index_dfs[index_name] = plot_df

    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout(pad=0.6, h_pad=1.8, w_pad=1.2)
    return fig, index_dfs


# -----------------------
# I/O helpers
# -----------------------

def _save_figure(fig, result_path, stem):
    """Save figure as PNG, PDF and SVG."""
    for ext in ("pdf", "svg"):
        path = result_path / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none",
                    format=ext)
        # print(f"Figure saved to: {path}")


def _save_index_data(index_dfs, result_path, prefix):
    """Save all per-index DataFrames to a single tidy CSV.

    Columns: plot_tool, index, target, sources, method, value
    """
    rows = []
    for index_name, df in index_dfs.items():
        tagged = df[["plot_tool", "target", "sources", "method", index_name]].copy()
        tagged["index"] = index_name
        tagged = tagged.rename(columns={index_name: "value"})
        rows.append(tagged)

    if not rows:
        return

    combined = pd.concat(rows, ignore_index=True)[
        ["plot_tool", "index", "target", "sources", "method", "value"]
    ]
    data_path = result_path / f"{prefix}_data.csv"
    combined.to_csv(data_path, index=False)
    # print(f"Data saved to: {data_path}")


# -----------------------
# Pipeline entry points
# -----------------------



def tool_comparison_plots(exp_name, exp_data, result_data, rerun):
    """Generate tool comparison plots for each dataset."""
    result_path = get_output_path() / f"{exp_name}/_results/{result_data.get('name')}"
    result_path.mkdir(parents=True, exist_ok=True)
    
    # indices = result_data.get("indices",None)
    # if indices is None:
    #     indices = INDICES_LIST

    for d in result_data.get("datasets", []):
        print(f"\nProcessing dataset: {d['name']}")

        tool_data = {}
        for t in TOOLS.keys():
            tool_data[t] = get_indices(exp_name, d["name"], t, d["tools"][t],rerun)

        fig, index_dfs = create_index_plots(
            tools_data=tool_data,
            indices=INDICES,
            width_cm=21,
            height_cm=9.9,
            n_cols_max=4,
            dpi=300
        )

        if fig is None:
            print(f"Warning: No figure generated for dataset {d['name']}")
            continue

        _save_index_data(index_dfs, result_path, prefix=d["name"])
        _save_figure(fig, result_path, stem=d["name"])
        plt.close(fig)


def tool_comparison_plots_combined(exp_name, exp_data, result_data, rerun):
    """Generate a single combined plot across all (or selected) datasets.

    Parameters
    ----------
    datasets : list of str
        Dataset names to include. Empty list (default) means all datasets.
    """
    result_path = get_output_path() / f"{exp_name}/_results/{result_data.get('name')}"
    result_path.mkdir(parents=True, exist_ok=True)

    all_datasets = result_data.get("datasets", [])
    all_dataset_tool_data = []
    
    for d in all_datasets:
        print(f"\nLoading dataset: {d['name']}")
        tool_data = {}
        for t in TOOLS.keys():
            tool_data[t] = get_indices(exp_name, d["name"], t, d["tools"][t],rerun)
        all_dataset_tool_data.append(tool_data)

    combined_tools_data = combine_dataset_tool_data(all_dataset_tool_data)
    suffix = "all"

    fig, index_dfs = create_index_plots(
        tools_data=combined_tools_data,
        indices=INDICES,
        width_cm=21,
        height_cm=9.9,
        n_cols_max=4,
        dpi=300
    )

    if fig is None:
        print("Warning: No figure generated for combined datasets")
        return

    _save_index_data(index_dfs, result_path, prefix=f"combined_{suffix}")
    _save_figure(fig, result_path, stem=f"combined_{suffix}")
    plt.close(fig)



RESULT_REGISTRY = {
    "tool_comparison_plots":          tool_comparison_plots,
    "tool_comparison_plots_combined": tool_comparison_plots_combined,
}


def get_exp_file(id):
    """Load experiment configuration file."""
    file_path = get_exp_path() / f"{id}.json"
    data = json.loads(file_path.read_text())
    return data


# -----
# CLI
# -----

def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--id",    required=True, help="experiment name")
    parent_parser.add_argument("--name",  required=True, help="result id")
    parent_parser.add_argument("--rerun", action="store_true", help="whether to rerun the experiment")

    parser = argparse.ArgumentParser(
        description="Generate result",
        parents=[parent_parser]
    )
    args = parser.parse_args()

    if args.id and args.name:
        exp    = get_exp_file(args.id)
        result = next(filter(lambda d: d.get("name") == args.name, exp.get("results", [])), None)
        if result is None:
            raise ValueError("Result not found")
        if result.get("type") not in RESULT_REGISTRY:
            raise ValueError("Invalid result type")
        RESULT_REGISTRY[result.get("type")](args.id, exp, result, args.rerun)
    else:
        raise ValueError("provide id and name")


if __name__ == "__main__":
    main()