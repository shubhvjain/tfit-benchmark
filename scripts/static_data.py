#!/usr/bin/env python
"""To generate tf lists  and other static data """

import json
import os
import sys
from pathlib import Path

import decoupler as dc
import pandas as pd

# Project imports
# PROJECT_ROOT = os.path.abspath("..")
# SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)
# if SCRIPTS_DIR not in sys.path:
#     sys.path.insert(0, SCRIPTS_DIR)

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
    # 1) Get CollecTRI network and targets with >= min_tf_per_target TFs
    net = dc.op.collectri(organism=organism)
    t = (
        net.groupby("target")["source"]
        .nunique()
        .reset_index()
        .rename(columns={"source": "count"})
    )
    collectri_list = t.loc[t["count"] >= min_tf_per_target, "target"].tolist()
    print(len(collectri_list))
    # 2) Datasets and TF list
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
    tf = set(e.load_tflist({}))

    result = {}
    common_set = set(collectri_list)

    for ds in datasets:
        print(ds)
        _, d = get_dataset(ds)
        all_genes = set(d.columns.tolist())
        targets = all_genes - tf  # remove TF genes to keep target-only genes

        # top_n_per_dataset most expressed target genes
        top_targets = filter_top_expressed(d, list(targets), top_n_per_dataset)
        result[ds] = list(   set(top_targets) &  set(collectri_list)  )
        print(len(result[ds]))
        # accumulate common genes with collectri_list
        common_set &= set(top_targets)

    output = {
        "params":{"collectri_min_tf_per_target":min_tf_per_target,"top_n_per_dataset":top_n_per_dataset },
        "datasets": result,
        "common": sorted(common_set),
        "collectri_list": sorted(collectri_list),
    }

    out_path = Path(out_path)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
   
    base_path = Path(os.getenv("DATA_STATIC_PATH"))
    print(base_path)
    #generate_collectri_list(out_path=base_path/"tf_collectri_20.json")
