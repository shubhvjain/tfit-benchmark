import argparse
from pathlib import Path
import os
import json
import pandas as pd

# from util import load_env
from gene_expression import Generate_Reports


def get_file(id):
    file_path = Path(os.getenv("ANALYSIS_INPUT_PATH"))/f"{id}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found")
    with open(file_path, "r") as f:
        file_data = json.load(f)

    # also make sure result output folder exists
    output_dir = Path(os.getenv("ANALYSIS_OUTPUT_PATH"))/f"{id}"
    os.makedirs(output_dir, exist_ok=True)

    return file_data

def gtex_stats(id, config):
    """
    generate report for gene expression data from GTEx portal
    """
    output_dir = Path(os.getenv("ANALYSIS_OUTPUT_PATH"))/f"{id}"
    Generate_Reports(dataset_list=config["datasets"],rerun_plots=config["rerun_plot"],rerun_summary=config["rerun_summary"],output_dir=output_dir)
    


RUN_REGISTRY = {
    "gtex_stats":gtex_stats
}


def run(id):
    # load_env()
    file = get_file(id)
    # print(file)
    if not file["type"] in RUN_REGISTRY.keys():
        raise ValueError("Invalid config.type")
    RUN_REGISTRY[file["type"]](id, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run result generation script")
    parser.add_argument("action", choices=[
                        "run"], help="Action to perform: 'new' to run pipeline or 'result' to generate clusters")
    parser.add_argument(
        "id", help="Result config filename (which is also the unique id)")

    args = parser.parse_args()

    if args.action == "run":
        run(args.id)
