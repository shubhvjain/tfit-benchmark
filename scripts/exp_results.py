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

from util import TOOLS,  get_exp_path,get_temp_path,get_output_path,get_data_path,get_dataset


INDICES_LIST = ["goa_lin_similarity","goa_resnik_similarity","goa_jc_similarity","shortest_PPI_path_score_hippie","shortest_PPI_path_score_stringdb","shortest_PPI_path_score_biogrid","shared_PPI_partners_score_hippie","shared_PPI_partners_score_stringdb","shared_PPI_partners_score_biogrid","grn_collectri"]

INDICES = {
    "goa_lin_similarity": {
        "title": "GO Similarity (Lin)",
        "plot":"box"
    },
    "goa_resnik_similarity": {
        "title": "GO Similarity (Resnik)",
        "plot":"box"
    },
    "goa_jc_similarity": {
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
FONT_SIZE_TICK    = 9  # tick labels
FONT_SIZE_PANEL   = 14    # panel letter  (A), (B) …
FONT_Y_LABEL = 12
# Seaborn palette – muted, grey-friendly, print-safe
_PALETTE = ["#058ED9","#f4ebd9", "#483d3f",  "#a39a92", "#77685d"]


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


def _pretty_label(index_name: str) -> str:
    """Turn snake_case index name into a short readable y-label."""
    # Strip trailing database suffixes for brevity
    for suffix in ("_hippie", "_stringdb", "_biogrid", "_collectri"):
        if index_name.endswith(suffix):
            index_name = index_name[: -len(suffix)]
            break
    return index_name.replace("_", " ").title()


# ----------------------
# Data preparation
# ---------------------

def prepare_combined_dataframe(tools_data):
    """Prepare combined dataframe with common targets."""
    # Get all unique targets from each tool
    all_targets = [set(df['target'].unique()) for df, _ in tools_data.values()]
    
    # Find common targets across ALL tools
    common_targets = set.intersection(*all_targets)
    
    # Create final dataframe combining all tools data for common targets
    final_dfs = []
    
    for tool_name, (df, _) in tools_data.items():
        # Filter for common targets
        df_filtered = df[df['target'].isin(common_targets)].copy()
        # Add plot_tool column
        df_filtered['plot_tool'] = tool_name
        final_dfs.append(df_filtered)
    
    # Combine all dataframes
    final_df = pd.concat(final_dfs, ignore_index=True)
    
    return final_df, common_targets

# -----------------------
# Single-panel drawing
# -----------------------

def create_index_subplot(ax, final_df, tools_data, index_name, plot_type, panel_label, y_label=None):
    """
    Draw one subplot panel using seaborn for publication-quality output.

    Parameters
    ----------
    ax          : matplotlib Axes
    final_df    : combined DataFrame (all tools, common targets)
    tools_data  : dict  {tool_name: (df, meta_dict)}
    index_name  : str   column to visualise
    plot_type   : str   'box' | 'histogram' | 'line' | 'none'
    panel_label : str   e.g. 'A', 'B', …
    """

    # Keep original keys for filtering
    tool_keys = list(tools_data.keys())
    # Create title mapping and order
    tool_order = [TOOLS[key]["title"] for key in tool_keys]
    
    # ── box 
    if plot_type == "box":
        # Per target keep only the row with the highest value for this index
        rows = []
        
        for tool_key in tool_keys:  # Use original keys here
            sub = final_df[final_df["plot_tool"] == tool_key]
            if sub.empty or index_name not in sub.columns:
                continue
            best = sub.loc[sub.groupby("target")[index_name].idxmax()]
            rows.append(best[["plot_tool", index_name]])

        if not rows:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    fontsize=FONT_SIZE_TICK, color="0.5", transform=ax.transAxes)
            ax.set_visible(True)
            return

        plot_df = pd.concat(rows, ignore_index=True)
        # NOW map to titles - after filtering with original keys
        plot_df["plot_tool"] = plot_df["plot_tool"].map(lambda x: TOOLS[x]["title"])
        
        #print(plot_df)
        sns.boxplot(
            data=plot_df,
            x="plot_tool",
            legend=False, 
            hue="plot_tool", 
            y=index_name,
            order=tool_order,  # This now matches the mapped titles
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

        # ax.set_xlabel("")
        # # print(y_label)
        # ax.set_ylabel(y_label or _pretty_label(index_name), fontsize=FONT_Y_LABEL,
        #               labelpad=2, fontweight="normal")
        # ax.tick_params(axis="x", labelsize=FONT_SIZE_TICK)
        # # Trim x tick labels if long
        # new_labels = []
        # for lbl in ax.get_xticklabels():
        #     t = lbl.get_text()
        #     new_labels.append(t[:12] if len(t) > 12 else t)
        # ax.set_xticklabels(new_labels, ha="center")

        # # Reduce y-tick density
        # ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune="both"))
        ax.set_xlabel("")
        ax.set_ylabel(y_label or _pretty_label(index_name), fontsize=FONT_Y_LABEL,
                    labelpad=2, fontweight="normal")
        ax.tick_params(axis="x", labelsize=FONT_SIZE_TICK)

        # Trim x tick labels if long - modify in place
        for lbl in ax.get_xticklabels():
            t = lbl.get_text()
            if len(t) > 12:
                lbl.set_text(t[:12])
            lbl.set_ha("center")

        # Reduce y-tick density
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune="both"))


    # ── line 
    elif plot_type == "line":
        for idx, tool_name in enumerate(tool_order):
            sub = final_df[final_df["plot_tool"] == tool_name]
            if sub.empty or index_name not in sub.columns:
                continue
            grouped = sub.groupby("target")[index_name].mean()
            ax.plot(range(len(grouped)), grouped.values,
                    marker="o", markersize=2,
                    label=tool_name,
                    color=_PALETTE[idx % len(_PALETTE)],
                    linewidth=0.9)

        ax.set_xlabel("Target", fontsize=FONT_SIZE_LABEL, labelpad=2)
        ax.set_ylabel(y_label, fontsize=6, labelpad=2)
        ax.legend(fontsize=FONT_SIZE_TICK, frameon=False,
                  loc="best", handlelength=1, handletextpad=0.4)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune="both"))

    # ── none 
    elif plot_type == "none":
        ax.set_visible(False)
        return

    else:
        ax.text(0.5, 0.5, f"unknown: {plot_type}", ha="center", va="center",
                fontsize=FONT_SIZE_TICK, color="red", transform=ax.transAxes)
        return

    _apply_base_style(ax)

    # Panel letter in the upper-left, outside the axes box
    ax.text(-0.18, 1.06, f"({panel_label})", transform=ax.transAxes,
            fontsize=FONT_SIZE_PANEL, fontweight="bold", va="top", ha="left",
            clip_on=False)


# ------------------------
# Multi-panel figure
# ------------------------

def create_index_plots(tools_data, indices, width_cm=7, height_cm=9.9, 
                       n_cols_max=4, dpi=300):
    """
    Create a publication-ready main image with subplots for each index.
    
    Parameters:
    -----------
    tools_data : dict
        Dictionary with structure {tool_name: (dataframe, dict), ...}
    indices : dict
        Dictionary with structure {index_name: {'title': 'A', 'plot': plot_type}, ...}
        where plot_type is 'box', 'histogram', 'line', or 'none'
    width_cm : float
        Width of the main figure in centimeters (default: 21 for A4 width)
    height_cm : float
        Height of the main figure in centimeters (default: 9.9 for 1/3 A4)
    n_cols_max : int
        Maximum number of columns (default: 4)
    dpi : int
        Resolution for saved figure (default: 300)
    
    Returns:
    --------
    fig : matplotlib figure
        The created figure object
    """
    # ── global rcParams ─────────────────────────────────────────────────────
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

    # ── layout ──────────────────────────────────────────────────────────────
    final_df, common_targets = prepare_combined_dataframe(tools_data)
    print(f"Common targets: {len(common_targets)}")

    plottable = {k: v for k, v in indices.items() if v.get("plot") != "none"}
    n_plots = len(plottable)
    if n_plots == 0:
        print("No plottable indices.")
        return None

    n_cols = min(n_cols_max, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    print(f"Layout: {n_rows} rows x {n_cols} cols")

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(width_cm, height_cm),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, (index_name, cfg) in enumerate(plottable.items()):
        panel_label = chr(65 + idx)  # A, B, C, D, ... (65 is ASCII for 'A')
        ax = axes_flat[idx]
        print(f"  ({panel_label}) {cfg['plot']}  ←  {index_name}")
        create_index_subplot(ax, final_df, tools_data,
                        index_name, cfg["plot"], panel_label,
                        y_label=cfg.get("title"))

    # Hide unused axes
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout(pad=0.6, h_pad=1.8, w_pad=1.2)
    return fig


# -----------------------
# Pipeline entry points  
# -----------------------


def tool_comparison_plots(exp_name, exp_data, result_data, rerun):
    """Generate tool comparison plots for each dataset."""
    result_path = get_output_path()/f"{exp_name}/_results/{result_data.get('name')}"
    result_path.mkdir(parents=True, exist_ok=True)
    
    for d in result_data.get("datasets", []):
        print(f"\nProcessing dataset: {d['name']}")
        
        # First get the score data and generate if required
        tool_data = {}
        for t in TOOLS.keys():
            tool_data[t] = get_indices(exp_name, d["name"], t, d["tools"][t], rerun)

        # Then generate the image for the dataset
        fig = create_index_plots(
            tools_data=tool_data,
            indices=INDICES,
            width_cm=21,      # A4 width
            height_cm=9.9,    # 1/3 A4 height
            n_cols_max=4,     # Maximum 4 columns
            dpi=300
        )
        
        if fig is None:
            print(f"Warning: No figure generated for dataset {d['name']}")
            continue
        
        # Save the figure
        save_path = result_path/f"{d['name']}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Figure saved to: {save_path}")
        
        plt.close(fig)  # Close to free memory


def compute_score(data, indices):
    """Compute scores for the given data."""
    if data is None:
        raise ValueError("No data provided")
    mask = data['sources'].str.split(";").str.len() > 1
    data = data[mask].copy()
    bio_data_path = Path(os.path.expandvars(os.getenv("DATA_PATH")))
    df_score, add_data = compute_indices(df=data, methods=indices, data_path=bio_data_path)
    return df_score, add_data


def get_indices(exp_name, dataset, tool, result_file_name, rerun=False):
    """Generate indices not already generated for tool results if not already generated and returns the values"""
    result_file = get_output_path()/f"{exp_name}/{dataset}/{tool}/{result_file_name}.csv"
    indices_file_csv = get_output_path()/f"{exp_name}/{dataset}/{tool}/{result_file_name}_score.csv"
    indices_file_json = get_output_path()/f"{exp_name}/{dataset}/{tool}/{result_file_name}_score.json"
    
    generate_scores = False

    if indices_file_csv.exists() and indices_file_json.exists():
        score_file = pd.read_csv(indices_file_csv)
        with open(indices_file_json, "r") as f:
            score_file_json = json.load(f) 
        if rerun:
            generate_scores = True
    else:
        generate_scores = True
    
    if generate_scores:
        score_file_raw = pd.read_csv(result_file)
        score_file, score_file_json = compute_score(score_file_raw, INDICES_LIST)
        score_file.to_csv(indices_file_csv, index=False)
        with open(indices_file_json, "w") as f:
            json.dump(score_file_json, f, indent=2) 
        
    return score_file, score_file_json


RESULT_REGISTRY = {
    "tool_comparison_plots": tool_comparison_plots
}


def get_exp_file(id):
    """Load experiment configuration file."""
    file_path = get_exp_path()/f"{id}.json"
    data = json.loads(file_path.read_text())
    return data


# -----
# CLI
# -----

def main():
    # Parent parser for common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--id", required=True, help="experiment name")
    parent_parser.add_argument("--name", required=True, help="result id")
    parent_parser.add_argument("--rerun", action="store_true", help="whether to rerun the experiment")
    
    parser = argparse.ArgumentParser(
        description="Generate result",
        parents=[parent_parser]  # Inherit arguments from parent
    )
    args = parser.parse_args()  # Parse into Namespace object

    if args.id and args.name: 
        exp = get_exp_file(args.id)
        result = next(filter(lambda d: d.get("name") == args.name, exp.get("results", [])), None)
        if result is None:
            raise ValueError("Result not found")
        
        if not result.get("type") in RESULT_REGISTRY.keys():
            raise ValueError("Invalid result type")

        RESULT_REGISTRY[result.get("type")](args.id, exp, result, args.rerun)

    else:
        raise ValueError("provide id and name")


if __name__ == "__main__":
    main()