#!/usr/bin/env python3
"""
Gene Cluster Performance Index Pipeline
============================
Three steps:
  1. init     — Load a CSV and distribute genes into bucket JSON files
  2. run      — Compute an index for all genes across all buckets
  3. collect  — Flatten completed bucket results into a final CSV

Usage:
  python pipeline.py init    --input data.csv [--buckets 50] [--bucket-dir buckets/]
  python pipeline.py run     --index index1 [--force] [--workers 4] [--bucket-dir buckets/]
  python pipeline.py collect --index index1,index2 [--output results.csv] [--bucket-dir buckets/]
"""

import os
import json
import argparse
import hashlib
import shutil
import time
import traceback
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from tfitpy import compute_indices
import numpy as np
import pandas as pd

from util import get_exp_path,get_temp_path,get_output_path



# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------


def _bucket_path(bucket_dir: Path, bucket_id: int) -> Path:
    return bucket_dir / f"bucket_{bucket_id:02d}.csv"

# ---------------------------------------------------------------------------
# Step 1: init
# ---------------------------------------------------------------------------

def cmd_init(exp=None, dataset=None, tool="coregtor", result_name=None, n_buckets=10, rerun=False):
    if exp is None or dataset is None or tool is None or result_name is None:
        raise ValueError("Missing data")

    input_dir  = get_output_path() / f"{exp}/{dataset}/{tool}"
    input_file = f"result_{result_name}"
    input_path = input_dir / f"{input_file}.csv"

    if not input_path.exists():
        print(input_path)
        raise ValueError("file not found")

    bucket_dir = get_temp_path() / f"{exp}/{dataset}/{tool}/{input_file}"
    bucket_dir.mkdir(parents=True, exist_ok=True)

    status_file = bucket_dir / "status.json"
    if status_file.exists() and not rerun:
        print("folder already setup. skipping")
        return

    df = pd.read_csv(input_path)

    required = {"cluster_uid", "target", "sources"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    df = pd.read_csv(input_path)
    chunks = np.array_split(df, n_buckets)

    written = 0
    for bid, chunk in enumerate(chunks):
        path = _bucket_path(bucket_dir, bid)
        chunk.to_csv(path, index=False)
        written += 1

    status = {
        "n_buckets":  n_buckets
    }
    status_file.write_text(json.dumps(status, indent=1))

    print(f"Done. {written} buckets written.")
    print(f"Total genes: {df['target'].nunique()}, buckets: {n_buckets}, dir: {bucket_dir}")

# ---------------------------------------------------------------------------
# Step 2: run
# ---------------------------------------------------------------------------


def compute_score(data):
    """Compute scores for the given data."""
    if data is None:
        raise ValueError("No data provided")
    mask = data['sources'].str.split(";").str.len() > 1
    data = data[mask].copy()
    bio_data_path = Path(os.path.expandvars(os.getenv("DATA_PATH")))
    df_score, add_data = compute_indices(df=data,  data_path=bio_data_path)
    return df_score, add_data


def create_progress_file(bucket_dir, bucket_id):
    (bucket_dir / f"bucket_{bucket_id:02d}.inprogress").touch()

def delete_progress_file(bucket_dir, bucket_id):
    p = bucket_dir / f"bucket_{bucket_id:02d}.inprogress"
    if p.exists():
        p.unlink()

def cmd_run_bucket(exp=None, dataset=None, tool="coregtor", result_name=None, bucket_id=None, rerun=False):
    if exp is None or dataset is None or tool is None or result_name is None or bucket_id is None:
        raise ValueError("Missing data")

    input_file = f"result_{result_name}"
    bucket_dir = get_temp_path() / f"{exp}/{dataset}/{tool}/{input_file}"

    status_file = bucket_dir / "status.json"
    if not status_file.exists():
        raise ValueError("Folder not setup. Run init first.")

    bucket_path = _bucket_path(bucket_dir, bucket_id)
    if not bucket_path.exists():
        raise ValueError(f"Bucket {bucket_id} not found")

    progress_file = bucket_dir / f"bucket_{bucket_id:02d}.inprogress"
    if progress_file.exists() and not rerun:
        raise ValueError("Bucket is already being processed. Use rerun=True to force.")

    try:
        data = pd.read_csv(bucket_path)
        create_progress_file(bucket_dir, bucket_id)
        new_data, _ = compute_score(data)
        new_data.to_csv(bucket_path, index=False)
    except Exception as e:
        print(f"Error processing bucket {bucket_id}: {e}")
        raise e
    finally:
        delete_progress_file(bucket_dir, bucket_id)


# ---------------------------------------------------------------------------
# Step 3: collect
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gene cluster index pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- init --
    p_init = sub.add_parser("init", help="Initialise bucket files from a CSV")
    p_init.add_argument("--id",      required=True,       help="exp name")
    p_init.add_argument("--dataset",      required=True,       help="dataset name")
    p_init.add_argument("--tool",      required=False, default="coregtor",   help="tool name")
    p_init.add_argument("--result",      required=True,       help="result name")
    p_init.add_argument("--n_buckets",    type=int, default=10, help="Number of buckets (default: 20)")
    p_init.add_argument("--rerun",      action="store_true", help="Recompute even if already done")

    # -- run --

    p_run = sub.add_parser("run", help="Compute an index across for a bucket")
    p_run.add_argument("--id",      required=True,       help="exp name")
    p_run.add_argument("--dataset",      required=True,       help="dataset name")
    p_run.add_argument("--tool",      required=False, default="coregtor",   help="tool name")
    p_run.add_argument("--result",      required=True,       help="result name")
    p_run.add_argument("--bucket",    type=int,   required=True,  help=" Buckets ID")
    p_run.add_argument("--rerun",      action="store_true", help="Recompute even if already done")


    # -- collect --
    p_col = sub.add_parser("collect", help="Flatten results into CSV files")
    p_col.add_argument("--index",         required=True,          help="Comma-separated index names to include")
    p_col.add_argument("--bucket-dir",    default="buckets",      help="Bucket directory (default: buckets/)")
    p_col.add_argument("--output",        default="results.csv",  help="Output CSV path (default: results.csv)")
    p_col.add_argument("--force-collect", action="store_true",    help="Export even if some indexes are incomplete")

    args = parser.parse_args()

    if args.command == "init":
        print(args.n_buckets)
        cmd_init(args.id,args.dataset,args.tool,args.result,args.n_buckets,args.rerun)
    elif args.command == "run":
        cmd_run_bucket(args.id,args.dataset,args.tool,args.result,args.bucket,args.rerun)
    elif args.command == "collect":
        cmd_collect(args)


if __name__ == "__main__":
    main()