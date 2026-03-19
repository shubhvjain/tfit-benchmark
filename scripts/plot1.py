import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
import matplotlib.colors as mcolors
def create_performance_scores_plot(ax=None, data=None, plot_options=None):
    """
    Create performance scores box plots for multiple indices.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes, optional
        The axes object to plot on. If None, creates new figure
    data : DataFrame
        DataFrame with index columns to plot
    plot_options : dict
        Configuration options
        
    Returns:
    --------
    ax : matplotlib.axes.Axes
    """
    
    # Index metadata
    INDEX_META = {
        "goa_similarity_lin": "GO Similarity (Lin)",
        "goa_similarity_resnik": "GO Similarity (Resnik)",
        "goa_similarity_jc": "GO Similarity (JC)",
        "shortest_PPI_path_score_hippie": "PPI Shortest Path (Hippie)",
        "shortest_PPI_path_score_stringdb": "PPI Shortest Path (StringDB)",
        "shortest_PPI_path_score_biogrid": "PPI Shortest Path (Biogrid)",
        "shared_PPI_partners_score_hippie": "PPI Shared Partners (Hippie)",
        "shared_PPI_partners_score_stringdb": "PPI Shared Partners (StringDB)",
        "shared_PPI_partners_score_biogrid": "PPI Shared Partners (Biogrid)",
        "grn_collectri_precision": "GRN Precision (Collectri)",
        "grn_collectri_recall": "GRN Recall (Collectri)",
        "grn_collectri_jaccard": "GRN Jaccard (Collectri)",
    }
    
    defaults = {
        'indices': None,
        'group_col': 'tool',
        'group_labels': {"coregtor":"Coregtor","coregnet":"CoRegNet"},
        'n_cols': 6,
        'palette': ["#00b0be", "#ffb255", "#ff8ca1", "#a39a92", "#77685d"],
        'figsize': [16, 10],
        'dpi': 300,
        'show_ylabel': True,
        'ylabel_fontsize': 8,
        'tick_fontsize': 7,
        'rotate_xlabels': True,
    }
    
    options = {**defaults, **(plot_options or {})}
    
    if data is None or data.empty:
        raise ValueError("Data required")
    
    # Determine indices to plot
    if options['indices'] is None:
        indices = INDEX_META.keys()
    else:
        indices = options['indices']
    
    n_indices = len(indices)
    n_cols = options['n_cols']
    n_rows = (n_indices + n_cols - 1) // n_cols
    
    # Create figure if needed
    if ax is None:
        fig_size = options.get("figsize")
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_size[0], fig_size[1]), dpi=options['dpi'])
        axes = np.atleast_2d(axes).flatten()
        standalone = True
    else:
        fig = ax.figure
        gs = ax.get_gridspec()
        ax.remove()
        inner_gs = gs[ax.get_subplotspec()].subgridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)
        axes = [fig.add_subplot(inner_gs[i]) for i in range(n_rows * n_cols)]
        standalone = False
    
    # Apply group labels if provided
    plot_df = data.copy()
    if options['group_labels'] is not None:
        plot_df[options['group_col']] = plot_df[options['group_col']].map(
            lambda x: options['group_labels'].get(x, x)
        )
    
    groups = plot_df[options['group_col']].unique()
    
    # Plot each index
    for idx, index_name in enumerate(indices):
        ax_i = axes[idx]
        
        # Check if we have valid data for this index
        valid_data = plot_df[[options['group_col'], index_name]].dropna()
        # Check if we have valid data for this index
        if len(valid_data) == 0:
            ax_i.set_visible(False)  # Hide instead of showing "No data"
            continue
        
        # Create base boxplot without palette
        sns.boxplot(
            data=valid_data,
            x=options['group_col'],
            y=index_name,
            ax=ax_i,
            width=0.25,
            linewidth=0.1,
            flierprops=dict(marker='o', markerfacecolor='0.4', 
                          markeredgewidth=0, markersize=1.5, alpha=0.5),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=0.8),      # And this for the caps
        )
        
        # Manually color the boxes based on number of groups present
        # num_groups = len(valid_data[options['group_col']].unique())
        # for patch_idx, patch in enumerate(ax_i.patches):
        #     color = options['palette'][patch_idx % len(options['palette'])]
        #     patch.set_facecolor(color)
        #     patch.set_edgecolor(color) 
        
        # for line_idx, line in enumerate(ax_i.lines):
        #   box_idx = line_idx // 6  # Each box has 6 lines (2 whiskers, 2 caps, 1 median, 1 flier)
        #   color = options['palette'][box_idx % len(options['palette'])]
        #   line.set_color(color)

        # After creating the boxplot and coloring the patches:
        for patch_idx, patch in enumerate(ax_i.patches):
            color = options['palette'][patch_idx % len(options['palette'])]
            patch.set_facecolor(color)
            
            # Make edge color darker
            darker_color = mcolors.to_rgb(color)
            darker_color = tuple(max(0, c - 0.2) for c in darker_color)
            patch.set_edgecolor(darker_color)

        # Also update the whiskers, caps, and medians:
        for line_idx, line in enumerate(ax_i.lines):
            box_idx = line_idx // 6
            color = options['palette'][box_idx % len(options['palette'])]
            darker_color = mcolors.to_rgb(color)
            darker_color = tuple(max(0, c - 0.2) for c in darker_color)
            line.set_color(darker_color)
        
        ax_i.set_xlabel('')
        
        ylabel = INDEX_META.get(index_name, index_name)
        if options['show_ylabel']:
            ax_i.set_ylabel(ylabel, fontsize=options['ylabel_fontsize'])
        else:
            ax_i.set_ylabel('')
        
        ax_i.tick_params(labelsize=options['tick_fontsize'], length=2, width=0.5)
        
        ax_i.spines['top'].set_visible(False)
        ax_i.spines['right'].set_visible(False)
        ax_i.spines['left'].set_linewidth(0.6)
        ax_i.spines['bottom'].set_linewidth(0.6)
        
        ax_i.grid(axis='y', color='0.85', linewidth=0.4, linestyle='--', zorder=0)
        ax_i.set_axisbelow(True)
        
        if options['rotate_xlabels'] and len(groups) > 2:
            for lbl in ax_i.get_xticklabels():
                lbl.set_rotation(0)
                lbl.set_ha('center')
                if len(lbl.get_text()) > 12:
                    lbl.set_text(lbl.get_text()[:12])
    
    for idx in range(n_indices, len(axes)):
        axes[idx].set_visible(False)
    
    if standalone:
        plt.tight_layout()
    
    return axes[0] if len(axes) > 0 else ax


EXP_PLOTS = {
    "create_performance_scores_plot": create_performance_scores_plot
}