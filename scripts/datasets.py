#!/usr/bin/env python3
"""
Download datasets defined in datasets.json.

Usage:
    python dataset.py --all
    python dataset.py --id amygdala
    python dataset.py --list
    python dataset.py --all --force
"""

import json
import os
import argparse
import sys
from pathlib import Path

import pooch


def load_catalogue() -> list[dict]:
    path = Path(__file__).parent.parent / "datasets.json"
    with open(path) as f:
        return json.load(f)["datasets"]


def get_data_root() -> Path:
    data_path = os.getenv("DATA_PATH")
    if not data_path:
        raise EnvironmentError("DATA_PATH env var not set")
    return Path(data_path)


def is_downloaded(dataset_id: str, data_root: Path) -> bool:
    d = data_root / dataset_id
    return d.exists() and any(f.is_file() for f in d.iterdir())


def download_dataset(dataset: dict, data_root: Path, force: bool = False):
    dataset_id = dataset["id"]
    dataset_dir = data_root / dataset_id

    if not force and is_downloaded(dataset_id, data_root):
        print(f"skip {dataset_id}")
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)

    url = dataset["url"]
    file_name = dataset["file_name"]
    known_hash = dataset.get("sha256")
    processor = pooch.Decompress(name=file_name) if url.endswith(".gz") else None

    print(f"downloading {dataset_id}")
    pooch.retrieve(
        url=url,
        known_hash=known_hash,
        fname=Path(url).name,
        path=dataset_dir,
        progressbar=True,
        processor=processor,
    )


def download_all(force: bool = False):
    data_root = get_data_root()
    datasets = load_catalogue()
    ok, failed = 0, []
    for ds in datasets:
        try:
            download_dataset(ds, data_root, force=force)
            ok += 1
        except Exception as e:
            print(f"failed {ds['id']}: {e}")
            failed.append(ds["id"])
    print(f"\n{ok}/{len(datasets)} successful")
    if failed:
        print(f"failed: {', '.join(failed)}")


def download_one(dataset_id: str, force: bool = False):
    data_root = get_data_root()
    datasets = load_catalogue()
    ds = next((d for d in datasets if d["id"] == dataset_id), None)
    if ds is None:
        print(f"'{dataset_id}' not found. available: {', '.join(d['id'] for d in datasets)}")
        sys.exit(1)
    download_dataset(ds, data_root, force=force)


def list_datasets():
    data_root = get_data_root()
    datasets = load_catalogue()
    for ds in datasets:
        status = "ok" if is_downloaded(ds["id"], data_root) else "missing"
        print(f"{ds['id']:<25} {status}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",  action="store_true")
    group.add_argument("--id",   metavar="ID")
    group.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if args.all:
        download_all(force=args.force)
    elif args.id:
        download_one(args.id, force=args.force)
    elif args.list:
        list_datasets()


if __name__ == "__main__":
    main()