#!/usr/bin/env python
"""To generate tf lists and other static data """

import json
import os
import sys
from pathlib import Path

import decoupler as dc
import pandas as pd

# Project imports
from util import get_dataset
import exp_init as e


def filter_top_expressed(df: pd.DataFrame, gene_list: list[str], n: int) -> list[str]:
    """Return top n genes from gene_list ranked by mean, then variance."""
    available = gene_list
    if not available:
        return []
    sub = df[available]
    stats = pd.DataFrame({"mean": sub.mean(), "variance": sub.var()})
    return (
        stats.sort_values(["mean", "variance"], ascending=False)
        .head(n)
        .index.tolist()
    )


def generate_collectri_list(
    organism: str = "human",
    min_tf_per_target: int = 20,
    top_n_per_dataset: int = 10000,
    out_path: str | Path = "tflist.json",
):
    # 1) Get CollecTRI network
    net = dc.op.collectri(organism=organism)
    
    # 2) Get targets with >= min_tf_per_target TFs (TARGET LIST)
    t = (
        net.groupby("target")["source"]
        .nunique()
        .reset_index()
        .rename(columns={"source": "count"})
    )
    collectri_targets = t.loc[t["count"] >= min_tf_per_target, "target"].tolist()
    print(f"CollecTRI targets with >={min_tf_per_target} TFs: {len(collectri_targets)}")
    
    # 3) Get all TFs (sources) that regulate these targets (TF LIST)
    collectri_net_filtered = net[net["target"].isin(collectri_targets)]
    collectri_tfs = collectri_net_filtered["source"].unique().tolist()
    print(f"TFs that regulate these targets: {len(collectri_tfs)}")
    
    # 4) Process datasets
    datasets = [
        "amygdala",
        "bladder",
        "blood",
        "cortex-ba9",
        "heart_left_ventricle",
        "heart_atrial_appendage",
        "hypothalamus",
        "kidney_cortex",
        "lung",
        "muscle_skeletal",
        "thyroid",
    ]
    
    known_tf_set = set(e.load_tflist({}))
    
    result_targets = {}
    result_tfs = {}
    common_targets = set(collectri_targets)
    common_tfs = set(collectri_tfs)

    for ds in datasets:
        print(f"\nProcessing {ds}...")
        _, d = get_dataset(ds)
        all_genes = set(d.columns.tolist())
        
        # Separate into targets and TFs based on known TF list
        dataset_targets = all_genes - known_tf_set
        dataset_tfs = all_genes & known_tf_set
        
        # Find which collectri targets are in this dataset (top expressed)
        top_targets = filter_top_expressed(d, list(dataset_targets), top_n_per_dataset)
        present_targets = set(top_targets) & set(collectri_targets)
        result_targets[ds] = sorted(present_targets)
        
        # Find which collectri TFs are in this dataset
        present_tfs = dataset_tfs & set(collectri_tfs)
        result_tfs[ds] = sorted(present_tfs)
        
        print(f"  Targets present: {len(present_targets)}")
        print(f"  TFs present: {len(present_tfs)}")
        
        # Update common sets
        common_targets &= present_targets
        common_tfs &= present_tfs

    print(f"\n=== Summary ===")
    print(f"Common targets across all datasets: {len(common_targets)}")
    print(f"Common TFs across all datasets: {len(common_tfs)}")

    output = {
        "params": {
            "collectri_min_tf_per_target": min_tf_per_target,
            "top_n_per_dataset": top_n_per_dataset
        },
       # "datasets_targets": result_targets,
       # "datasets_tfs": result_tfs,
        "common_targets": sorted(common_targets),
        "common_tfs": sorted(common_tfs),
        #"collectri_targets": sorted(collectri_targets),
        #"collectri_tfs": sorted(collectri_tfs),
    }

    out_path = Path(out_path)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path.resolve()}")


if __name__ == "__main__":
    base_path = Path(os.getenv("DATA_STATIC_PATH"))
    print(base_path)
    generate_collectri_list(out_path=base_path/"collectri_20.json")