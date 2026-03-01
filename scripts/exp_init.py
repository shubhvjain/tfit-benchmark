#!/usr/bin/env python3
"""
Experiment initialization and status checking.

Run before any container or pipeline step.

Usage:
    python exp_init.py init exp1
    python exp_init.py status exp1
    python exp_init.py status exp1 --dataset amygdala --tool coregtor
"""

import json
import os
import sys
import sqlite3
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from util import  get_exp_path,get_temp_path,get_output_path,get_data_path,get_dataset


# ------ Load exp and related files ------

def load_exp(exp_name: str) -> dict:
    filepath = get_exp_path() / f"{exp_name}.json"
    if not filepath.exists():
        raise FileNotFoundError(f"exp file not found: {filepath}")
    with open(filepath) as f:
        return json.load(f)

def load_tflist(exp: dict) -> list[str]:
    tf_cfg = exp.get("tf", {})
    if "custom_file_path" in tf_cfg:
        path = Path(os.path.expandvars(tf_cfg["custom_file_path"]))
        df = pd.read_csv(path)
        return df["gene_name"].tolist()
    data_root = get_data_path()
    meta_path = data_root / "tflist" / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    tf_path = data_root / "tflist" / meta["file_name"]
    df = pd.read_csv(tf_path, names=["gene_name"], header=None)
    return df["gene_name"].tolist()


# ------ Gene selection ------

def filter_top_expressed(df: pd.DataFrame, gene_list: list[str], n: int) -> list[str]:
    available = [g for g in gene_list if g in df.columns]
    sub = df[available]
    stats = pd.DataFrame({"mean": sub.mean(), "variance": sub.var()})
    return stats.sort_values(["mean", "variance"], ascending=False).head(n).index.tolist()


def filter_non_zero(df: pd.DataFrame, gene_list: list[str], threshold: float) -> list[str]:
    available = [g for g in gene_list if g in df.columns]
    counts = (df[available] > 0).sum() / len(df)
    return counts[counts >= threshold].index.tolist()


def resolve_gene_lists(exp: dict, dataset: pd.DataFrame, tflist: list[str]) -> tuple[list, list]:
    tf_cfg = exp.get("tf", {})
    target_cfg = exp.get("target", {})

    tf_in_data = set(tflist) & set(dataset.columns)

    # resolve tf list
    if "top_expressed_n" in tf_cfg:
        final_tf = filter_top_expressed(dataset, list(tf_in_data), tf_cfg["top_expressed_n"])
    elif "non_zero_threshold" in tf_cfg:
        final_tf = filter_non_zero(dataset, list(tf_in_data), tf_cfg["non_zero_threshold"])
    else:
        final_tf = list(tf_in_data)

    # resolve target list
    target_type = target_cfg.get("type", "no_tf")
    tf_set = set(final_tf)

    if target_type == "all":
        base_targets = set(dataset.columns)
    elif target_type == "no_tf":
        base_targets = set(dataset.columns) - tf_set
    elif target_type == "tf_only":
        base_targets = tf_set
    elif target_type == "custom":
        items = target_cfg.get("items", [])
        if not items:
            raise ValueError("target.type=custom requires target.items")
        base_targets = set(items) & set(dataset.columns)
    else:
        raise ValueError(f"unknown target.type: {target_type}")

    if "top_expressed_n" in target_cfg:
        final_targets = filter_top_expressed(dataset, list(base_targets), target_cfg["top_expressed_n"])
    elif "non_zero_threshold" in target_cfg:
        final_targets = filter_non_zero(dataset, list(base_targets), target_cfg["non_zero_threshold"])
    else:
        final_targets = list(base_targets)
    return list(set(final_tf)), list(set(final_targets))


# ------ input.json file generation (one per dataset per experiment)------

def generate_input_json(exp_name: str, dataset_id: str, exp: dict) -> dict:
    temp_dir = get_temp_path() / exp_name / dataset_id
    input_path = temp_dir / "input.json"

    if input_path.exists():
        print(f"  input.json exists: {dataset_id}")
        with open(input_path) as f:
            return json.load(f)

    print(f"  generating input.json: {dataset_id}")
    _,dataset = get_dataset(dataset_id)
    tflist = load_tflist(exp)
    final_tf, final_targets = resolve_gene_lists(exp, dataset, tflist)

    input_data = {
        "exp_id": exp_name,
        "dataset_id": dataset_id,
        "tf": final_tf,
        "targets": final_targets,
        "stats": {
            "tf_count": len(final_tf),
            "target_count": len(final_targets)
        },
        "exp_file_modified_time": int(os.path.getmtime( get_exp_path()/f"{exp_name}.json"))
    }

    temp_dir.mkdir(parents=True, exist_ok=True)
    with open(input_path, "w") as f:
        json.dump(input_data, f)
    
    # also create a copy in the output folder
    opt_dir = get_output_path() / exp_name / dataset_id
    opt_dir.mkdir(parents=True, exist_ok=True)
    with open(opt_dir/"input.json", "w") as f:
        json.dump(input_data, f)

    print(f"    tf: {len(final_tf)}, targets: {len(final_targets)}")
    return input_data


# ------ SQLite status db ------

def init_status_db(exp_name: str, dataset_id: str, tool: str, targets: list[str]):
    db_dir = get_output_path() / exp_name /  dataset_id / tool 
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "status.db"

    if db_path.exists():
        print(f"  status.db exists: {tool}/{dataset_id}")
        return db_path

    print(f"  creating status.db: {tool}/{dataset_id} ({len(targets)} genes)")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE genes (
            gene        TEXT PRIMARY KEY,
            status      TEXT DEFAULT 'pending',
            worker      TEXT,
            started_at  REAL,
            finished_at REAL,
            error       TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO genes (gene, status) VALUES (?, 'pending')",
        [(g,) for g in targets]
    )
    conn.commit()
    conn.close()
    print(f"    created with {len(targets)} pending genes")
    return db_path


# ------ Status ------

def get_status(exp_name: str, dataset_id: str = None, tool: str = None) -> dict:
    """
    Returns status for an experiment.
    If dataset_id and tool provided, returns gene-level counts.
    If only exp_name provided, returns summary across all datasets and tools.
    """
    output_root = get_output_path() / exp_name

    if not output_root.exists():
        return {"exp": exp_name, "status": "not initialized"}

    if tool and dataset_id:
        db_path = output_root / tool / dataset_id / "status.db"
        if not db_path.exists():
            return {"exp": exp_name, "dataset": dataset_id, "tool": tool, "status": "not initialized"}
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM genes GROUP BY status"
        ).fetchall()
        conn.close()
        counts = {row[0]: row[1] for row in rows}
        total = sum(counts.values())
        return {
            "exp": exp_name,
            "dataset": dataset_id,
            "tool": tool,
            "total": total,
            "pending": counts.get("pending", 0),
            "claimed": counts.get("claimed", 0),
            "done": counts.get("done", 0),
            "failed": counts.get("failed", 0)
        }

    # summary across all tools and datasets
    summary = {"exp": exp_name, "tools": {}}
    for tool_dir in sorted(output_root.iterdir()):
        if not tool_dir.is_dir():
            continue
        tool_name = tool_dir.name
        summary["tools"][tool_name] = {}
        for ds_dir in sorted(tool_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            db_path = ds_dir / "status.db"
            if not db_path.exists():
                continue
            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM genes GROUP BY status"
            ).fetchall()
            conn.close()
            counts = {row[0]: row[1] for row in rows}
            total = sum(counts.values())
            done = counts.get("done", 0)
            summary["tools"][tool_name][ds_dir.name] = {
                "total": total,
                "done": done,
                "pending": counts.get("pending", 0),
                "failed": counts.get("failed", 0),
                "pct": round(100 * done / total, 1) if total else 0
            }
    return summary


# ------ Init ------ 

def init_experiment(exp_name: str):
    print(f"initializing experiment: {exp_name}")
    exp = load_exp(exp_name)

    datasets = exp.get("datasets", [])
    tools = exp.get("tools", [])

    if not datasets:
        raise ValueError("no datasets specified in exp file")
    if not tools:
        raise ValueError("no tools specified in exp file")

    for dataset_id in datasets:
        print(f"\ndataset: {dataset_id}")
        input_data = generate_input_json(exp_name, dataset_id, exp)
        targets = input_data["targets"]
        for tool in tools:
            init_status_db(exp_name, dataset_id, tool, targets)

    print(f"\ndone")


# ------ CLI ------

def main():
    parser = argparse.ArgumentParser(description="Initialize and check experiment status")
    subparsers = parser.add_subparsers(dest="action", required=True)

    init_p = subparsers.add_parser("init")
    init_p.add_argument("exp_name")

    status_p = subparsers.add_parser("status")
    status_p.add_argument("exp_name")
    status_p.add_argument("--dataset", default=None)
    status_p.add_argument("--tool", default=None)

    args = parser.parse_args()

    if args.action == "init":
        init_experiment(args.exp_name)
    elif args.action == "status":
        result = get_status(args.exp_name, args.dataset, args.tool)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()