"""
Co-regulator identification through clustering of gene context similarity matrices.

Provides a unified interface for multiple clustering methods with consistent output format.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples
from scipy.cluster.hierarchy import linkage, inconsistent

from scipy.spatial.distance import squareform
from coregtor.utils.error import CoRegTorError
import secrets


def _ensure_distance_matrix(sim_matrix: pd.DataFrame) -> np.ndarray:
    """Convert similarity matrix to validated distance matrix.

    Args:
      sim_matrix: Input similarity matrix DataFrame (square).

    Returns:
      Validated distance matrix as numpy array.
    """
    is_distance = sim_matrix.attrs.get('is_distance', False)
    if not is_distance:
        dist = 1.0 - sim_matrix.values
    else:
        dist = sim_matrix.values.copy()
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)  # remove floating point negatives
    return dist.astype(np.float64)


def _compute_auto_threshold(sim_matrix, method, linkage1):
    """Compute automatic distance threshold using specified method.

    Methods:
    - 'inconsistency': Statistical jumps in dendrogram [μ(coeffs) + 0.5]
    - 'elbow': Largest gap in last 10 merge heights 
    - 'percentile': 75th percentile of all pairwise distances (fastest)

    Args:
      sim_matrix: Similarity matrix DataFrame
      method: {'inconsistency', 'elbow', 'percentile'}
      linkage1: Linkage method ('average', 'complete', etc.)

    Returns:
      float: Optimal distance_threshold
    """

    dist_matrix = _ensure_distance_matrix(sim_matrix)
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method=linkage1)

    if method == 'inconsistency':
        """Cut where dendrogram has statistical jumps.
        inconsistent(Z) → (n-1,4) array, col3=inconsistency coeffs (>1=odd merge)
        threshold = mean(coeffs) + 0.5*std(coeffs)
        """
        incons = inconsistent(Z)
        coeffs = incons[:, 3]  # Inconsistency coefficients
        return np.mean(coeffs) + 0.5 * np.std(coeffs)

    elif method == 'elbow':
        """Find largest gap in merge distances (elbow plot for dendrogram).
        Z[:,-10:,2] = last 10 merge heights [0.12,0.23,...,0.89←jump!,1.02]
        diffs = np.diff(heights), threshold = heights[max_diff_idx]
        """
        merge_heights = Z[-min(10, len(Z)):, 2]
        # merge_heights = Z[-min(10, len(Z)), 2]
        if len(merge_heights) < 2:
            return np.median(squareform(dist_matrix))
        diffs = np.diff(merge_heights)
        elbow_idx = np.argmax(diffs)
        return merge_heights[elbow_idx]

    elif method == 'percentile':
        """75th percentile of all pairwise distances (fast, robust).
        squareform(dist) → [0.12,0.45,0.89,...], cuts "large" distances
        """
        return np.percentile(squareform(dist_matrix), 75)

    else:
        raise ValueError(f"Unknown auto_threshold method: {method}. "
                         f"Use 'inconsistency', 'elbow', or 'percentile'")


def _format_clusters_df(
    sim_matrix: pd.DataFrame,
    cluster_labels: np.ndarray,
    target_gene: str,
    min_module_size: int = 2
) -> pd.DataFrame:
    dist_matrix = _ensure_distance_matrix(sim_matrix)
    gene_names = sim_matrix.index.astype(str)
    n_samples = len(gene_names)
    unique_labels = np.unique(cluster_labels)
    n_unique = len(unique_labels)

    valid_for_silhouette = 2 <= n_unique <= n_samples - 1
    if not valid_for_silhouette:
        sil_per_gene = np.zeros(n_samples)
    else:
        sil_per_gene = silhouette_samples(
            dist_matrix, cluster_labels, metric='precomputed')

    # Build gene-level frame
    gene_df = pd.DataFrame({
        'gene': gene_names,
        'cluster_id': cluster_labels,
        'sil': sil_per_gene
    })

    # Filter small clusters
    cluster_sizes = gene_df['cluster_id'].value_counts()
    valid_clusters = cluster_sizes[cluster_sizes >= min_module_size].index
    gene_df = gene_df[gene_df['cluster_id'].isin(valid_clusters)].copy()

    if gene_df.empty:
        return pd.DataFrame(columns=['cluster_uid', 'target', 'sources', 'n_sources', 'n_percent', 'silhouette_score'])

    # Aggregate to one row per cluster
    rows = []
    for cluster_id, group in gene_df.groupby('cluster_id'):
        genes = group.sort_values('sil', ascending=False)['gene'].tolist()
        n_sources = len(genes)
        sil_score = 0.0 if not valid_for_silhouette else round(
            group['sil'].mean(), 4)
        rows.append({
            'cluster_uid': secrets.token_hex(6),
            'target': target_gene,
            'sources': ';'.join(genes),
            'n_sources': n_sources,
            'silhouette_score': sil_score,
        })

    return pd.DataFrame(rows).sort_values('silhouette_score', ascending=False).reset_index(drop=True)

def _compute_threshold_for_min_cluster_size(
    sim_matrix: pd.DataFrame,
    min_cluster_size: int,
    linkage1: str,
    optimization: str = 'first'
) -> Optional[float]:
    """Find distance threshold ensuring at least one cluster has >= min_cluster_size.
    
    Args:
        sim_matrix: Similarity matrix DataFrame
        min_cluster_size: Minimum size for at least one cluster
        linkage1: Linkage method ('average', 'complete', etc.)
        optimization: Strategy for choosing among valid thresholds:
            - 'first': First threshold that satisfies constraint (largest clusters)
            - 'silhouette': Maximize silhouette score among valid thresholds
            - 'fewest': Minimize number of clusters (most aggressive merging)
    
    Returns:
        Optimal distance threshold, or None if constraint cannot be satisfied
    """
    from scipy.cluster.hierarchy import fcluster
    
    dist_matrix = _ensure_distance_matrix(sim_matrix)
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method=linkage1)
    
    distances = Z[:, 2]  # Merge heights
    
    # Try thresholds from min to max
    valid_thresholds = []
    valid_clusters = []
    valid_scores = []
    
    n_samples = len(sim_matrix)
    test_thresholds = np.linspace(distances.min(), distances.max(), 100)
    
    for threshold in test_thresholds:
        clusters = fcluster(Z, threshold, criterion='distance')
        unique, counts = np.unique(clusters, return_counts=True)
        
        # Check if at least one cluster meets size requirement
        if np.any(counts >= min_cluster_size):
            valid_thresholds.append(threshold)
            valid_clusters.append(clusters)
            
            # Calculate score for optimization
            if optimization == 'silhouette':
                n_unique = len(unique)
                if 2 <= n_unique <= n_samples - 1:
                    from sklearn.metrics import silhouette_score
                    score = silhouette_score(dist_matrix, clusters, metric='precomputed')
                else:
                    score = -1.0
                valid_scores.append(score)
            elif optimization == 'fewest':
                valid_scores.append(-len(unique))  # Negative for minimization
    
    if not valid_thresholds:
        return None  # Cannot satisfy constraint
    
    # Choose optimal threshold based on strategy
    if optimization == 'first':
        return valid_thresholds[0]
    elif optimization in ['silhouette', 'fewest']:
        best_idx = np.argmax(valid_scores)
        return valid_thresholds[best_idx]
    else:
        raise ValueError(f"Unknown optimization: {optimization}")

def _select_optimal_cluster(
    clusters_df: pd.DataFrame,
    selection_method: str = 'silhouette'
) -> Optional[Dict[str, Any]]:
    """Select optimal cluster from results.
    
    Args:
        clusters_df: DataFrame with all clusters
        selection_method: Method to select best cluster
            - 'silhouette': Highest average silhouette score (default)
            - 'largest': Most genes
            - 'first': First cluster (highest silhouette after sort)
    
    Returns:
        Dict with cluster info or None if no clusters
    """
    if clusters_df.empty:
        return None
    
    # If only one cluster, return it regardless of method
    if len(clusters_df) == 1:
        best_row = clusters_df.iloc[0]
    elif selection_method == 'silhouette':
        best_row = clusters_df.loc[clusters_df['silhouette_score'].idxmax()]
    elif selection_method == 'largest':
        best_row = clusters_df.loc[clusters_df['n_sources'].idxmax()]
    elif selection_method == 'first':
        best_row = clusters_df.iloc[0]
    else:
        raise ValueError(f"Unknown selection_method: {selection_method}")
    
    return best_row.to_dict()

def hierarchical_clustering(
    sim_matrix: pd.DataFrame,
    target_gene: str,
    method_options: Dict[str, Any],
    **kwargs
) -> Tuple[Optional[Any], pd.DataFrame]:
    """Perform hierarchical clustering with fixed parameters.

    Args:
      sim_matrix: Gene similarity matrix DataFrame (square, rows/cols = gene names).
      target_gene: Target gene name (must be in sim_matrix.index).
      method_options: Dict with:
        - 'n_clusters': int (exact number of clusters) OR 
        - 'distance_threshold': float (cut dendrogram at distance) OR
        - 'auto_threshold': str ['inconsistency', 'elbow', 'percentile'] OR
        - 'min_cluster_size': int (ensure at least one cluster has this many genes)
        - 'linkage': str ['average', 'complete', 'ward', 'single'] (default: 'average')
        - 'min_module_size': int (default: 2) - filter out small clusters from results
        - 'cluster_size_optimization': str ['first', 'silhouette', 'fewest'] 
            (only used with min_cluster_size, default: 'first')
      **kwargs: Ignored.

    Raises:
      CoRegTorError: If multiple conflicting clustering parameters specified.

    Returns:
      Tuple of (model, clusters_df)
    """
    n_clusters = method_options.get('n_clusters', None)
    distance_threshold = method_options.get('distance_threshold', None)
    auto_threshold = method_options.get('auto_threshold', None)
    min_cluster_size = method_options.get('min_cluster_size', None)
    linkage = method_options.get('linkage', 'average')
    min_module_size = method_options.get('min_module_size', 2)
    cluster_size_optimization = method_options.get('cluster_size_optimization', 'first')

    # Handle edge case: very small matrices
    if sim_matrix.shape[0] <= 3:
        gene_names = sim_matrix.index.astype(str).tolist()
        single_row = pd.DataFrame([{
            'cluster_uid': secrets.token_hex(6),
            'target': target_gene,
            'sources': ';'.join(gene_names),
            'n_sources': len(gene_names),
            'silhouette_score': 0.0,
        }])
        optimal_cluster = single_row.iloc[0].to_dict() if not single_row.empty else None
        return None, single_row, optimal_cluster
        # return None, single_row

    # Count how many clustering methods specified
    methods_specified = sum([
        n_clusters is not None,
        distance_threshold is not None,
        auto_threshold is not None,
        min_cluster_size is not None
    ])

    if methods_specified == 0:
        raise CoRegTorError(
            "Must specify one of: 'n_clusters', 'distance_threshold', "
            "'auto_threshold', or 'min_cluster_size'"
        )
    
    if methods_specified > 1:
        raise CoRegTorError(
            "Cannot specify multiple clustering methods. Choose only one of: "
            "'n_clusters', 'distance_threshold', 'auto_threshold', or 'min_cluster_size'"
        )

    # Compute distance_threshold based on the specified method
    if auto_threshold:
        distance_threshold = _compute_auto_threshold(
            sim_matrix, auto_threshold, linkage
        )
    elif min_cluster_size:
        distance_threshold = _compute_threshold_for_min_cluster_size(
            sim_matrix, min_cluster_size, linkage, cluster_size_optimization
        )
        if distance_threshold is None:
            raise CoRegTorError(
                f"Cannot find threshold satisfying min_cluster_size={min_cluster_size}. "
                f"Maximum possible cluster size is {len(sim_matrix)}."
            )

    # Perform clustering
    dist_matrix = _ensure_distance_matrix(sim_matrix)
    params = {'metric': 'precomputed', 'linkage': linkage}
    
    if n_clusters is not None:
        params['n_clusters'] = n_clusters
        params['distance_threshold'] = None
    else:
        params['n_clusters'] = None
        params['distance_threshold'] = distance_threshold

    model = AgglomerativeClustering(**params)
    labels = model.fit_predict(dist_matrix)

    # Format results
    clusters_df = _format_clusters_df(
        sim_matrix, labels, target_gene, min_module_size
    )

    optimal_cluster = _select_optimal_cluster(
        clusters_df, 
        method_options.get('cluster_selection', 'silhouette')
    )

    return model, clusters_df, optimal_cluster
    #return model, clusters_df

# def validation_index(
#     sim_matrix: pd.DataFrame,
#     target_gene: str,
#     method_options: Dict[str, Any],
#     **kwargs
# ) -> Dict[str, Any]:
#   """Automatic cluster number selection using validation indices.

#   Args:
#     sim_matrix: Gene similarity matrix.
#     target_gene: Target gene name.
#     method_options: Parameters dict with 'index', 'k_range'.
#     **kwargs: Additional parameters.

#   Returns:
#     Dict with model, clusters_df, best, best_df, methodology, validation_scores.
#   """
#   index = method_options.get('index', 'silhouette')
#   linkage1 = method_options.get('linkage', 'average')
#   k_range = method_options.get('k_range', (2, 20))
#   min_module_size = method_options.get('min_module_size', 2)

#   dist_matrix = _ensure_distance_matrix(sim_matrix)
#   n_samples = dist_matrix.shape[0]

#   condensed_dist = squareform(dist_matrix)
#   Z = linkage(condensed_dist, method=linkage1)

#   scores = {}
#   k_max = min(k_range[1] + 1, n_samples)
#   for k in range(max(2, k_range[0]), k_max):
#     labels = fcluster(Z, k, criterion='maxclust')

#     try:
#       if index == 'silhouette':
#         score = silhouette_score(dist_matrix, labels, metric='precomputed')
#       elif index == 'davies_bouldin':
#         score = davies_bouldin_score(dist_matrix, labels, metric='precomputed')
#       elif index == 'calinski_harabasz':
#         score = calinski_harabasz_score(dist_matrix, labels)
#       else:
#         raise ValueError(f"Unknown index: {index}")
#     except:
#       score = -1.0 if index == 'silhouette' else float('inf')

#     scores[k] = score

#   if not scores:
#     raise CoRegTorError(f"No valid k values in range {k_range}")

#   if index == 'silhouette':
#     best_k = max(scores.keys(), key=lambda k: scores[k])
#     if scores[best_k] <= 0:
#       best_k = None
#   elif index == 'davies_bouldin':
#     best_k = min(scores.keys(), key=lambda k: scores[k])
#   elif index == 'calinski_harabasz':
#     best_k = max(scores.keys(), key=lambda k: scores[k])

#   if best_k is None:
#     raise CoRegTorError(f"No valid clustering found using {index}")

#   labels = fcluster(Z, best_k, criterion='maxclust')
#   clusters_df = _format_clusters_df(sim_matrix, labels, target_gene, min_module_size)

#   best_cluster = get_best_cluster(clusters_df)


#   method_options_best = method_options.copy()
#   method_options_best['best_k'] = best_k

#   best_methodology = format_methodology("validation_index", method_options_best)

#   best = {
#       'items': best_row['gene_cluster'].split(';'),
#       'score': best_score,
#       'methodology': best_methodology
#   }

#   best_df = pd.DataFrame([{
#       'target_gene': target_gene,
#       'cluster_id': best_row['cluster_id'],
#       'items': best_row['gene_cluster'],
#       'score': best_score,
#       'methodology': best_methodology
#   }])

#   methodology = format_methodology("validation_index", method_options_best)

#   return {
#       'model': None,
#       'clusters_df': clusters_df,
#       'best': best,
#       'best_df': best_df,
#       'methodology': methodology,
#       'validation_scores': scores
#   }


METHOD_REGISTRY = {
    'hierarchical_clustering': hierarchical_clustering,
    # 'validation_index': validation_index
}


def identify_coregulators(
    sim_matrix: pd.DataFrame,
    target_gene: str,
    method: str = "hierarchical_clustering",
    method_options: Dict[str, Any] = {},
    **kwargs
):
    """Identify co-regulatory modules from gene similarity matrix.

    Args:
      sim_matrix: Similarity matrix DataFrame from context comparison.
      target_gene: Target gene identifier.
      method: Clustering method name. (hierarchical_clustering,validation_index)
      method_options: Method-specific parameters dictionary.

    Returns:
      Dict containing:
        - model: Fitted clustering model or None
        - clusters_df: DataFrame of all clusters
        - best: Best cluster information dict or None
        - best_df: Best cluster as single-row DataFrame or None
        - methodology: Complete parameter string
        - validation_scores: Validation scores dict (validation_index only)
    """
    if method not in METHOD_REGISTRY:
        available = list(METHOD_REGISTRY.keys())
        raise CoRegTorError(
            f"Unknown method '{method}'. Available: {available}")

    method_func = METHOD_REGISTRY[method]
    return method_func(sim_matrix, target_gene, method_options, **kwargs)
