"""
im using exp01 results for coregulators 
for now using the "default" method  of clustering  
"""
import pandas as pd
from pathlib import Path
import os


def get_output_path() -> Path:
    p = os.getenv("EXP_OUTPUT_PATH")
    if not p:
        raise EnvironmentError("EXP_OUTPUT_PATH not set")
    return Path(p)


def coregulators(options):
    """
    """
    results = []
    for d in options["dataset"]:
        result_file = get_output_path()/"exp01"/d/"coregtor" / \
            "result_default_clusters.csv"
        if result_file.exists():
            dataset = pd.read_csv(result_file)
            for g in options["gene"]:
                print()
                result = dataset[dataset['target'] == g]
                if not result.empty:
                    row_list = result.iloc[0].to_dict()
                    row_list["sources"] = sorted(
                        row_list["sources"].split(";"))
                    # row_list["sources"] = row_list["sources"].sort()
                    # print("Found:", row_list)
                    results.append({"success": True, "type": "coregulator_network", "data": row_list,
                                   "title": f"Coregulators of gene {g} (derived from {d} dataset)"})
                else:
                    print("Row not found")
                    results.append(
                        {"success": False, "message": f"Gene {g} not found in dataset {d}"})
        else:
            results.append(
                {"success": False, "message": f"Results not yet available for dataset {d}"})

    return results


def coregulator_clusters(options):
    """
    """
    results = []

    for d in options["dataset"]:
        result_file = get_output_path() / "exp01" / d / "coregtor" / \
            "result_default_indices.csv"
        

        if result_file.exists():
            dataset = pd.read_csv(result_file)
            for g in options["gene"]:
                gene_data = []  # Array of dicts for this gene
                result = dataset[dataset['target'] == g]

                if not result.empty:
                    gene_data = []
                    for idx in range(len(result)):
                        row_list = result.iloc[idx].to_dict()
                        row_list = {k: (None if pd.isna(v) else v)
                                    for k, v in row_list.items()}
                        #row_list["sources"] = sorted(row_list["sources"].split(";"))

                        gene_data.append(row_list)
                    results.append({
                        "success": True,
                        "type": "clusters_comparison",
                        "data": gene_data,
                        "title": f"Potential Coregulating Clusters of gene {g} in dataset {d}"
                    })
                
                else:
                    # print("clusters not found")
                    results.append(
                        {"success": False, "message": f"Clusters for gene {g} not found in dataset {d}"})
        else:
            results.append(
                {"success": False, "message": f"Results not yet available for dataset {d}"})

    return results
