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
import numpy as np

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
    "grn_precision_recall_collectri": {
        "title": "D",
        "plot":"line"
    },
    "grn_set_metrics_collectri": {
        "title": "E",
        "plot":"none"
    },
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


def tool_dataset_index_plot():
    """
    """


def tool_dataset_plot(exp_name, exp_data, dataset_details,result_path):
    """
    """



def tool_comparison_plots(exp_name,exp_data,result_data):
    """"""
    result_path = get_output_path()/f"{exp_name}/_results/{result_data.get("name")}"
    result_path.mkdir(parents=True,exist_ok=True)
    
    for d in result_data.get("datasets",[]) :
        print(d)
        for t in TOOLS.keys():
            get_indices(exp_name,d["name"],t,d["tools"][t])
        #tool_dataset_plot(exp_name,exp_data,d,result_path)

def compute_score(data,indices):
    """"""
    if data is None:
        raise ValueError("No data provided")
    mask = data['sources'].str.split(";").str.len() > 1
    data = data[mask].copy()
    bio_data_path = Path(os.path.expandvars(os.getenv("DATA_PATH")))
    df_score,add_data = compute_indices(df=data, methods=indices, data_path=bio_data_path)
    return df_score,add_data

# def _compute_performance(id, config, data, indices):
#     """
#     compute_performance generates the performance indices for the give dataframe of clusters generated. they are separated by a group_id

#     :param data: dataframe with 4 required cols: cluster_id, group_id, target, sources
#     """
#     if data is None:
#         raise ValueError("No data provided")

#     # remove fields with just one source
#     mask = data['sources'].str.split(";").str.len() > 1
#     data = data[mask].copy()

#     data_path = Path(os.path.expandvars(os.getenv("DATA_PATH")))
#     jobs = config.get("n_jobs", 2)
#     data_results = compute_indices(
#         data, data_path=data_path, n_jobs=jobs, methods=indices)
#     # print(data_results)
#     return data_results


def get_indices(exp_name,dataset,tool,result_file_name,rerun=False):
    """generate indices not already generated for tool results if not already generated and returns the values"""
    result_file = get_output_path()/f"{exp_name}/{dataset}/{tool}/{result_file_name}.csv"
    indices_file_csv = get_output_path()/f"{exp_name}/{dataset}/{tool}/{result_file_name}_score.csv"
    indices_file_json = get_output_path()/f"{exp_name}/{dataset}/{tool}/{result_file_name}_score.json"

    all_scores = set(INDICES.keys())
    
    generate_scores = False

    if indices_file_csv.exists() and indices_file_json.exists() :
        score_file = pd.read_csv(indices_file_csv)
        with open(indices_file_json,"r") as f:
            score_file_json = json.load(f) 
        if rerun:
            generate_scores = True
    else:
        generate_scores = True
    

    if generate_scores:
        score_file_raw = pd.read_csv(result_file)
        score_file,score_file_json  = compute_score(score_file_raw,INDICES.keys())
        score_file.to_csv(indices_file_csv)
        with open(indices_file_json,"w") as f:
            json.dump(score_file_json,f) 
        
    return score_file,score_file_json
        



RESULT_REGISTRY = {
    "tool_comparison_plots": tool_comparison_plots
}


def get_exp_file(id):
    """
    """
    file_path = get_exp_path()/f"{id}.json"
    data = json.loads(file_path.read_text())
    return data

# ------ CLI ------

def main():
    # Parent parser for common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--id", required=True, help="experiment name")
    parent_parser.add_argument("--name", required=True, help="result id")
    
    parser = argparse.ArgumentParser(
        description="Generate result",
        parents=[parent_parser]  # Inherit arguments from parent
    )
    args = parser.parse_args()  # Parse into Namespace object

    if args.id and args.name: 
        exp = get_exp_file(args.id)
        result = next(filter(lambda d: d.get("name") == args.name,exp.get("results",[])))
        if result is None:
            raise ValueError("Not found")
        
        if not result.get("type") in RESULT_REGISTRY.keys():
            raise ValueError("Invalid result type")

        RESULT_REGISTRY[result.get("type")](args.id,exp,result)

    else:
        raise ValueError("provide id and name")



    #plot(args.id, args.plot)  # Use parsed args

if __name__ == "__main__":
    main()