import pandas as pd
import numpy as np
import json
from util import get_experiment_paths,get_dataset
import sys
import os
import secrets
from pathlib import Path
import argparse

TOOL="coregnet"

def start_tool(exp_name,dataset_id,rerun=False):
    paths = get_experiment_paths(exp_name,dataset_id,TOOL)

    if not paths["input_file"].exists():
        print(f"input.json not found: {paths['input_file']}")
        print("run exp_init.py first")
        sys.exit(1)

    if not paths["db_file"].exists():
        print(f"status.db not found: {paths['db_file']}")
        print("run exp_init.py first")
        sys.exit(1)
    
    with open(paths["input_file"]) as f:
        input_data = json.load(f)

    with open(paths["exp_file"]) as f:
        exp = json.load(f)

    paths["temp_folder"].mkdir(parents=True, exist_ok=True)
    paths["output_folder"].mkdir(parents=True, exist_ok=True)

    _, dataset = get_dataset(dataset_id)


    return input_data, dataset


# def get_data(input_details: dict, dataset: pd.DataFrame):
#     """
#     Prepare expression matrix for CoRegNet.
#     Returns log2(x+1) transformed matrix (genes x samples) as numpy array,
#     plus tf and target gene lists.
#     """
#     tf_vec     = input_details["tf"]
#     target_vec = input_details["targets"]
#     genes      = tf_vec + target_vec

#     subset = dataset[genes]                      # samples x genes
#     matrix = np.log2(subset.values + 1)         # log2 transform
#     matrix = matrix.T                            # genes x samples

#     # Return as DataFrame so R gets row/col names automatically
#     result = pd.DataFrame(matrix, index=genes, columns=dataset.index)

#     return result, tf_vec, target_vec

def get_data(input_details: dict, dataset: pd.DataFrame):
    """
    Prepare expression matrix for CoRegNet.
    """
    tf_vec     = input_details["tf"]
    target_vec = input_details["targets"]
    
    # Get unique genes (order doesn't matter)
    genes = list(set(tf_vec + target_vec))
    
    subset = dataset[genes]  # samples x genes
    subset = subset.loc[:, ~subset.columns.duplicated(keep='first')]
    matrix = np.log2(subset.values + 1)         
    matrix = matrix.T                            
    result = pd.DataFrame(matrix, index=genes, columns=dataset.index)    
    return result, tf_vec, target_vec      



def coregnet_results(exp_name,dataset_id):
    
    grn_path = Path(os.getenv("EXP_OUTPUT_PATH")) / exp_name/ dataset_id / "coregnet" / "grn.csv"
    out_path = Path(os.getenv("EXP_OUTPUT_PATH")) / exp_name/ dataset_id / "coregnet" / "results.csv"

    if not grn_path.exists():
        raise FileNotFoundError(f"GRN file not found: {grn_path}")

    grn = pd.read_csv(grn_path)
    method_label = f"{exp_name}-coregnet"

    rows = []
    for target, group in grn.groupby("Target"):
        sources     = sorted(group["Regulator"].tolist())
        sources_str = ";".join(sources)
        n_sources   = len(sources)

        rows.append({
            "cluster_uid": secrets.token_hex(6),
            "target":      target,
            "sources":     sources_str,
            "n_sources":   n_sources,
            "method":      method_label
        })

    results = pd.DataFrame(rows)
    results.to_csv(out_path, index=False)
    print(f"Saved {len(results)} rows to {out_path}")



def main():
    # Parent parser for common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--id", required=True, help="experiment name")
    parent_parser.add_argument("--dataset", required=True, help="dataset id")
    
    parser = argparse.ArgumentParser(
        description="Generate coregnet results",
        parents=[parent_parser]  # Inherit arguments from parent
    )
    args = parser.parse_args()  # Parse into Namespace object

    coregnet_results(args.id, args.dataset)  # Use parsed args

if __name__ == "__main__":
    main()
