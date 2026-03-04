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
import numpy as np
from io import BytesIO

from tfitpy import compute_indices

from util import TOOLS,  get_exp_path,get_temp_path,get_output_path,get_data_path,get_dataset


INDICES = {
    "goa_lin_similarity": {
        "title": "A",
        "plot":"box"
    },
    "goa_resnik_similarity": {
        "title": "B",
        "plot":"box"
    },
    "goa_jc_similarity": {
        "title": "C",
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
    "shortest_PPI_path_score_hippie": {
        "title": "F",
        "plot":"box"
    },
     "shortest_PPI_path_score_stringdb": {
        "title": "G",
        "plot":"box"
    },
      "shortest_PPI_path_score_biogrid": {
        "title": "H",
        "plot":"box"
    },
      "shared_PPI_partners_score_hippie": {
        "title": "I",
        "plot":"box"
    },
      "shared_PPI_partners_score_stringdb": {
        "title": "J",
        "plot":"box"
    },
    "shared_PPI_partners_score_biogrid": {
        "title": "K",
        "plot":"box"
    }
}

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


def create_index_subplot(ax, final_df, tools_data, index_name, plot_type, panel_label):
    """
    Create a subplot for a single index with publication-ready styling.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis object to plot on
    final_df : DataFrame
        Combined dataframe with all tools data
    tools_data : dict
        Dictionary of tools data
    index_name : str
        Name of the index/column to plot
    plot_type : str
        Type of plot ('box', 'histogram', 'line', etc.)
    panel_label : str
        Label for the panel (e.g., 'A', 'B', 'C')
    """
    
    if plot_type == 'box':
        # For each tool, find the row with highest index score per target
        tool_max_data = []
        
        for tool_name in tools_data.keys():
            tool_df = final_df[final_df['plot_tool'] == tool_name]
            
            # For each target, find row with highest value for this index
            if len(tool_df) > 0 and index_name in tool_df.columns:
                max_rows = tool_df.loc[tool_df.groupby('target')[index_name].idxmax()]
                
                tool_max_data.append({
                    'tool': tool_name,
                    'values': max_rows[index_name].values
                })
        
        if not tool_max_data:
            ax.text(0.5, 0.5, f'No data for {index_name}', 
                   ha='center', va='center', fontsize=8, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # Prepare data for box plot
        box_data = [item['values'] for item in tool_max_data]
        labels = [item['tool'] for item in tool_max_data]
        
        # Create box plot with publication styling
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                        widths=0.6,
                        boxprops=dict(linewidth=1.2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        medianprops=dict(color='red', linewidth=1.5))
        
        # Customize colors - use publication-friendly palette
        colors = ['#E8F4F8', '#D4E9F7', '#B8D8EB', '#9AC7E0', '#7CB6D4']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
        
        ax.set_ylabel(index_name.replace('_', ' ').title(), fontsize=6, fontweight='normal')
        ax.tick_params(axis='both', labelsize=7)
        ax.tick_params(axis='x', rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        
    elif plot_type == 'histogram':
        # Create histogram for each tool
        n_tools = len(tools_data)
        # Publication-friendly colors (grayscale-friendly)
        colors = ['#4472C4', '#70AD47', '#FFC000', '#C00000', '#7030A0']
        
        for idx, tool_name in enumerate(tools_data.keys()):
            tool_df = final_df[final_df['plot_tool'] == tool_name]
            
            if len(tool_df) > 0 and index_name in tool_df.columns:
                ax.hist(tool_df[index_name], bins=8, alpha=0.6, 
                       label=tool_name, color=colors[idx % len(colors)], 
                       edgecolor='black', linewidth=0.8)
        
        ax.set_xlabel(index_name.replace('_', ' ').title(), fontsize=6, fontweight='normal')
        ax.set_ylabel('Frequency', fontsize=6, fontweight='normal')
        ax.legend(fontsize=6, frameon=False, loc='best')
        ax.tick_params(axis='both', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
    
    elif plot_type == 'line':
        # Line plot implementation
        colors = ['#4472C4', '#70AD47', '#FFC000', '#C00000', '#7030A0']
        
        for idx, tool_name in enumerate(tools_data.keys()):
            tool_df = final_df[final_df['plot_tool'] == tool_name]
            
            if len(tool_df) > 0 and index_name in tool_df.columns:
                # Group by target and plot
                grouped = tool_df.groupby('target')[index_name].mean()
                ax.plot(grouped.index, grouped.values, 
                       marker='o', label=tool_name, 
                       color=colors[idx % len(colors)], linewidth=1.5)
        
        ax.set_xlabel('Target', fontsize=8, fontweight='normal')
        ax.set_ylabel(index_name.replace('_', ' ').title(), fontsize=8, fontweight='normal')
        ax.legend(fontsize=6, frameon=False, loc='best')
        ax.tick_params(axis='both', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        
    elif plot_type == 'none':
        # Hide the subplot for 'none' type
        ax.set_visible(False)
        return
    
    else:
        # Unknown plot type - show warning
        ax.text(0.5, 0.5, f'Unknown plot type: {plot_type}', 
               ha='center', va='center', fontsize=8, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    # Add panel label (A), (B), (C), etc.
    ax.text(-0.15, 1.05, f'({panel_label})', transform=ax.transAxes,
           fontsize=10, fontweight='bold', va='top', ha='left')
    
    # Add light grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)


def create_index_plots(tools_data, indices, width_cm=21, height_cm=9.9, 
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
    
    # Convert cm to inches for matplotlib
    width_inch = width_cm / 2.54
    height_inch = height_cm / 2.54
    
    # Prepare combined dataframe
    final_df, common_targets = prepare_combined_dataframe(tools_data)
    
    print(f"Common targets found: {len(common_targets)}")
    
    # Filter out indices with plot type 'none'
    plottable_indices = {k: v for k, v in indices.items() if v.get('plot') != 'none'}
    
    print(f"Creating plots for {len(plottable_indices)} indices...")
    
    # Calculate subplot layout
    n_plots = len(plottable_indices)
    if n_plots == 0:
        print("No plottable indices found.")
        return None
    
    n_cols = min(n_cols_max, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
    
    print(f"Layout: {n_rows} rows × {n_cols} columns")
    
    # Set publication-ready style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    
    # Create main figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width_inch, height_inch))
    
    # Handle single subplot case
    if n_plots == 1:
        axes = np.array([axes])
    
    # Flatten axes array for easier iteration
    if n_plots > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = axes
    
    # Create subplot for each index with panel labels from the title field
    for idx, (index_name, config) in enumerate(plottable_indices.items()):
        plot_type = config['plot']
        panel_label = config.get('title', chr(65 + idx))  # Use title field or fallback to A, B, C...
        ax = axes_flat[idx]
        
        print(f"  ({panel_label}) Creating {plot_type} plot for: {index_name}")
        create_index_subplot(ax, final_df, tools_data, index_name, plot_type, panel_label)
    
    # Hide empty subplots if any
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # Adjust layout to prevent overlap - tighter for publication
    plt.tight_layout(pad=1.5, h_pad=2.5, w_pad=2.0)
    
    return fig


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
        
        # Add main title
        fig.suptitle(f"Tool Comparison: {d['name']}", 
                    fontsize=10, fontweight='bold', y=0.98)
        
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

    all_scores = set(INDICES.keys())
    
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
        score_file, score_file_json = compute_score(score_file_raw, INDICES.keys())
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


# ------ CLI ------

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