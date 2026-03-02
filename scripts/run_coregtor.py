#!/usr/bin/env python3
"""
Run coregtor pipeline for a given experiment, dataset.

Usage:
    python coregtor_run.py run exp1 amygdala
    python coregtor_run.py result exp1 amygdala
"""

import os
import sys
import json
import sqlite3
import argparse
from pathlib import Path
from util import get_dataset
from coregtor_pipeline import Pipeline, PipelineResults
from util import get_output_path,get_temp_path,get_exp_path,get_experiment_paths
TOOL = "coregtor"

#  ------  Config  ------ 

def build_config(exp_name: str, dataset_id: str, exp: dict, targets: list[str]) -> dict:
    paths = get_experiment_paths(exp_name, dataset_id,TOOL)
    config = {}
    config.update(exp.get("tool_run", {}).get(TOOL, {}))
    config.update(exp.get("tool_result", {}).get(TOOL, {}))
    config["target_genes"] = targets
    config["checkpointing"] = True
    config["paths"] = {
        "temp":   str(paths["temp_folder"]),
        "output": str(paths["output_folder"])
    }
    return config

#  ------  Status db  ------ 

def claim_pending_genes(db_path: Path, worker_id: str, batch_size: int = 500) -> list[str]:
    """Atomically claim a batch of pending genes. Returns claimed gene list."""
    import time
    STALE_TIMEOUT = 60 * 60 * 2  # 2 hours - adjust to your expected job duration
    
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    
    # Reset stale claimed genes (worker died/crashed)
    conn.execute(
        "UPDATE genes SET status='pending', worker=NULL WHERE status='claimed' AND started_at < ?",
        (time.time() - STALE_TIMEOUT,)
    )
    conn.commit()


    # get pending genes
    rows = conn.execute(
        "SELECT gene FROM genes WHERE status='pending' LIMIT ?",
        (batch_size,)
    ).fetchall()

    if not rows:
        conn.close()
        return []

    genes = [r[0] for r in rows]
    conn.executemany(
        "UPDATE genes SET status='claimed', worker=?, started_at=? WHERE gene=? AND status='pending'",
        [(worker_id, time.time(), g) for g in genes]
    )
    conn.commit()

    # confirm how many we actually claimed (race condition guard)
    claimed = conn.execute(
        f"SELECT gene FROM genes WHERE worker=? AND status='claimed'",
        (worker_id,)
    ).fetchall()
    conn.close()
    return [r[0] for r in claimed]


def mark_gene_done(db_path: Path, gene: str):
    import time
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "UPDATE genes SET status='done', finished_at=? WHERE gene=?",
        (time.time(), gene)
    )
    conn.commit()
    conn.close()


def mark_gene_failed(db_path: Path, gene: str, error: str):
    import time
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "UPDATE genes SET status='failed', finished_at=?, error=? WHERE gene=?",
        (time.time(), error[:500], gene)
    )
    conn.commit()
    conn.close()


def all_genes_done(db_path: Path) -> bool:
    conn = sqlite3.connect(db_path, timeout=30)
    row = conn.execute(
        "SELECT COUNT(*) FROM genes WHERE status != 'done'"
    ).fetchone()
    conn.close()
    return row[0] == 0


#  ------  Run  ------ 

def run(exp_name: str, dataset_id: str, worker_id: str, batch_size: int):
    paths = get_experiment_paths(exp_name, dataset_id,TOOL)
    #print(paths)
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

    # claim genes from db
    targets = claim_pending_genes(paths["db_file"], worker_id, batch_size)
    if not targets:
        print("no pending genes, nothing to do")
        return

    print(f"claimed {len(targets)} genes as worker {worker_id}")

    paths["temp_folder"].mkdir(parents=True, exist_ok=True)
    paths["output_folder"].mkdir(parents=True, exist_ok=True)

    config = build_config(exp_name, dataset_id, exp, targets)
    tfs = input_data["tf"]


    _, dataset = get_dataset(dataset_id)

    pipeline = Pipeline(
        expression_data=dataset,
        tflist=tfs,
        options=config,
        exp_title=f"{exp_name}_{dataset_id}"
    )

    # run sequentially, update db per gene
    for gene in targets:
        try:
            print(gene)
            pipeline.run_single_target(gene)
            mark_gene_done(paths["db_file"], gene)
            print(" done")
        except Exception as e:
            print("  error")
            print(e)
            mark_gene_failed(paths["db_file"], gene, str(e))

    print("done")


#  ------  Result  ------ 

def result(exp_name: str, dataset_id: str,n_jobs:int = 4):
    paths = get_experiment_paths(exp_name, dataset_id,TOOL)

    if not all_genes_done(paths["db_file"]):
        print("not all genes are done yet, check status.db")

    # Get list of successfully completed targets from database
    conn = sqlite3.connect(paths["db_file"], timeout=30)
    targets = conn.execute(
        "SELECT gene FROM genes WHERE status='done' ORDER BY gene"
    ).fetchall()
    conn.close()
    
    target_list = [r[0] for r in targets]
    print(f"Found {len(target_list)} successfully completed genes")

    with open(paths["input_file"]) as f:
        input_data = json.load(f)

    with open(paths["exp_file"]) as f:
        exp = json.load(f)

    config = build_config(exp_name, dataset_id, exp, [])
    if "result_generation" not in config:
        config["result_generation"]= {}
    config["result_generation"]["n_jobs"] = n_jobs
    tfs = input_data["tf"]

    resp = PipelineResults(
        options=config,
        exp_title=f"{exp_name}_{dataset_id}",
        tflist=tfs,
        targets=target_list
    )
    resp.generate_clusters_file()
    print("results generated")



def update_run_status(db_path: Path, checkpoint_dir: Path, check_internal: bool = True, n_jobs: int = 4, batch_size: int = 500):
    """Reconcile DB status with checkpoint files on disk.
    
    Args:
        db_path: Path to SQLite database
        checkpoint_dir: Directory containing .pkl checkpoint files
        check_internal: If True, read pkl files to check internal success/failure status
        n_jobs: Number of parallel workers for reading pkl files (when check_internal=True)
        batch_size: Number of files to process before committing to DB
    
    Behavior:
    - If check_internal=False (fast mode):
        - If .pkl exists -> mark as 'done'
        - If status is 'claimed' but no .pkl -> reset to 'pending'
    
    - If check_internal=True (thorough mode):
        - Reads each .pkl file to check internal success field
        - If success=True -> mark as 'done'
        - If success=False -> mark as 'run_failure' with error message
        - If status is 'claimed' but no .pkl -> reset to 'pending'
    """
    import joblib
    from joblib import Parallel, delayed

    def check_pkl_file(pkl_file: Path) -> tuple[str, str, str | None]:
        gene = pkl_file.stem
        try:
            checkpoint = joblib.load(pkl_file)
            status = checkpoint.get("success", {})
            if status.get("success", False):
                return ("done", gene, None)
            return ("run_failure", gene, (status.get("error", "") or "")[:500])
        except Exception as e:
            return ("run_failure", gene, f"Failed to read checkpoint: {str(e)[:400]}")

    def flush_batch(conn, results: list):
        done_genes = [(g,) for status, g, _ in results if status == "done"]
        failed_genes = [(err, g) for status, g, err in results if status == "run_failure"]

        if done_genes:
            conn.executemany(
                "UPDATE genes SET status='done', worker=NULL, error=NULL WHERE gene=? AND status != 'done'",
                done_genes,
            )
        if failed_genes:
            conn.executemany(
                "UPDATE genes SET status='run_failure', worker=NULL, error=? WHERE gene=?",
                failed_genes,
            )
        conn.commit()

    pkl_files = list(checkpoint_dir.glob("*.pkl"))
    completed_on_disk = {f.stem for f in pkl_files}
    total = len(pkl_files)

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")

    try:
        if not check_internal:
            print(f"Fast mode: marking {total} checkpoint files as done...")
            if completed_on_disk:
                conn.executemany(
                    "UPDATE genes SET status='done', worker=NULL, error=NULL WHERE gene=? AND status != 'done'",
                    [(g,) for g in completed_on_disk],
                )
                conn.commit()

        else:
            print(f"Thorough mode: checking {total} files in batches of {batch_size} with {n_jobs} workers...")

            batches = [pkl_files[i:i + batch_size] for i in range(0, total, batch_size)]

            for batch_idx, batch in enumerate(batches):
                results = Parallel(n_jobs=n_jobs, verbose=0)(
                    delayed(check_pkl_file)(f) for f in batch
                )
                flush_batch(conn, results)

                processed = min((batch_idx + 1) * batch_size, total)
                print(f"  Processed {processed}/{total} files...", end="\r")

            print()  # newline after progress line

        # Reset stale claimed jobs
        stale = conn.execute("SELECT gene FROM genes WHERE status='claimed'").fetchall()
        stale_genes = [(g,) for (g,) in stale if g not in completed_on_disk]
        if stale_genes:
            print(f"Resetting {len(stale_genes)} stale claimed genes to pending...")
            conn.executemany(
                "UPDATE genes SET status='pending', worker=NULL WHERE gene=?",
                stale_genes,
            )
            conn.commit()

        # Summary
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM genes GROUP BY status ORDER BY status"
        ).fetchall()
        print("\nStatus summary:")
        for status, count in rows:
            print(f"  {status}: {count}")

    finally:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()


def reset_failed(db_path: Path):
    """Reset all failed genes back to pending."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "UPDATE genes SET status='pending', worker=NULL, started_at=NULL, finished_at=NULL,  error=NULL WHERE status='failed'"
    )
    conn.commit()
    row = conn.execute("SELECT COUNT(*) FROM genes WHERE status='pending'").fetchone()
    conn.close()
    print(f"reset {row[0]} failed genes to pending")

def reset_claimed(db_path: Path, worker_id: str):
    """Reset all claimed genes by a specific worker back to pending."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "UPDATE genes SET status='pending', worker=NULL, started_at=NULL, finished_at=NULL  WHERE status='claimed' AND worker=?",
        (worker_id,)
    )
    conn.commit()
    row = conn.execute("SELECT COUNT(*) FROM genes WHERE status='pending'").fetchone()
    conn.close()
    print(f"reset genes claimed by '{worker_id}' to pending, total pending now: {row[0]}")


#  ------  CLI  ------ 

def main():
    # Parent parser for common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--id", required=True, help="experiment name")
    parent_parser.add_argument("--dataset", required=True, help="dataset id")
    parent_parser.add_argument("--n_jobs", type=int, default=8, help="number of parallel jobs")
    
    parser = argparse.ArgumentParser(description="Run coregtor pipeline")
    subparsers = parser.add_subparsers(dest="action", required=True)
    
    run_p = subparsers.add_parser("run", parents=[parent_parser])
    run_p.add_argument("--worker", default="worker_0", help="unique worker id for this job")
    run_p.add_argument("--batch", type=int, default=500, help="number of genes to claim per run")
    
    res_p = subparsers.add_parser("result", parents=[parent_parser])
    
    reset_p = subparsers.add_parser("reset_failed", parents=[parent_parser])
    
    rc_p = subparsers.add_parser("reset_claimed", parents=[parent_parser])
    rc_p.add_argument("--worker", required=True, help="worker id to reset")
    
    status_p = subparsers.add_parser("update_status", parents=[parent_parser])
    status_p.add_argument("--read", action="store_true", help="read pkl files to check internal success/failure (slower but more accurate)")
    
    args = parser.parse_args()
    
    if args.action == "run":
        run(args.id, args.dataset, args.worker, args.batch)
    elif args.action == "result":
        result(args.id, args.dataset, n_jobs=args.n_jobs)
    elif args.action == "update_status":
        paths = get_experiment_paths(args.id, args.dataset, TOOL)
        update_run_status(paths["db_file"], paths["temp_folder"], 
                         check_internal=args.read, n_jobs=args.n_jobs)
    elif args.action == "reset_failed":
        paths = get_experiment_paths(args.id, args.dataset, TOOL)
        reset_failed(paths["db_file"])
    elif args.action == "reset_claimed":
        paths = get_experiment_paths(args.id, args.dataset, TOOL)
        reset_claimed(paths["db_file"], args.worker)

if __name__ == "__main__":
    main()
