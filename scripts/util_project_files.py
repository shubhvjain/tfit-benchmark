#!/usr/bin/env python3
"""
Create and validate project files (experiments, analyses).

Usage:
    python project_files.py new exp exp1
    python project_files.py new analysis analysis1
    python project_files.py check exp exp1
    python project_files.py check analysis analysis1
"""

import json
import os
import sys
import argparse
from pathlib import Path
from jsonschema import validate, ValidationError, Draft7Validator


# Schemas 

SCHEMAS = {
    "exp": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["type", "datasets", "target", "tf", "tools", "tool_run", "tool_result"],
        "properties": {
            "type": {
                "type": "string",
                "enum": ["run_tool"],
                "default": "run_tool"
            },
            "about": {
                "type": "string",
                "default": ""
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "default": []
            },
            "rerun": {
                "type": "boolean",
                "default": False
            },
            "datasets": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["dataset_id"]
            },
            "target": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["all", "no_tf", "tf_only", "custom"]
                    },
                    "top_expressed_n": {"type": "integer"},
                    "non_zero_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                    "items": {"type": "array", "items": {"type": "string"}}
                },
                "default": {"type": "no_tf", "top_expressed_n": 20}
            },
            "tf": {
                "type": "object",
                "properties": {
                    "top_expressed_n": {"type": "integer"},
                    "non_zero_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                    "custom_file_path": {"type": "string"}
                },
                "default": {"top_expressed_n": 50}
            },
            "tools": {
                "type": "array",
                "items": {"type": "string", "enum": ["coregtor", "coregnet", "netrem"]},
                "default": ["coregtor"]
            },
            "tool_run": {
                "type": "object",
                "properties": {
                    "coregtor": {"type": "object"},
                    "coregnet": {"type": "object"},
                    "netrem": {"type": "object"}
                },
                "default": {
                    "coregtor": {
                        "create_model": {
                            "model": "rf",
                            "model_options": {
                                "max_depth": 5,
                                "n_estimators": 1000,
                                "random_state": 120
                            }
                        },
                        "tree_paths": {},
                        "create_context": {"method": "tree_paths"},
                        "transform_context": [
                            {
                                "id": "default",
                                "method": "gene_frequency",
                                "normalize": False,
                                "min_frequency": 1
                            }
                        ],
                        "compare_context": [
                            {
                                "id": "default",
                                "method": "cosine",
                                "transformation_id": "default",
                                "convert_to_distance": False
                            }
                        ]
                    }
                }
            },
            "tool_result": {
                "type": "object",
                "properties": {
                    "coregtor": {"type": "object"},
                    "coregnet": {"type": "object"},
                    "netrem": {"type": "object"}
                },
                "default": {
                    "coregtor": {
                        "clustering": [
                            {
                                "id": "default",
                                "matrix_id": "default",
                                "method": "hierarchical_clustering",
                                "method_options": {"auto_threshold": "inconsistency"}
                            }
                        ]
                    }
                }
            }
        }
    },

    "analysis": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["type", "experiments"],
        "properties": {
            "type": {
                "type": "string",
                "enum": ["result_comparison"],
                "default": "result_comparison"
            },
            "about": {
                "type": "string",
                "default": ""
            },
            "rerun": {
                "type": "boolean",
                "default": False
            },
            "experiments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["exp", "dataset", "tool", "result"],
                    "properties": {
                        "exp":     {"type": "string"},
                        "dataset": {"type": "string"},
                        "tool":    {"type": "string"},
                        "result":  {"type": "string"},
                        "title":   {"type": "string"}
                    }
                },
                "default": [
                    {
                        "exp": "exp_id",
                        "dataset": "dataset_id",
                        "tool": "coregtor",
                        "result": "result_default",
                        "title": ""
                    }
                ]
            },
            "n_jobs": {
                "type": "integer",
                "default": 4
            }
        }
    }
}


# Paths 

def get_path(filetype: str) -> Path:
    if filetype == "exp":
        p = os.getenv("EXP_INPUT_PATH")
        if not p:
            raise EnvironmentError("EXP_PATH env var not set")
    elif filetype == "analysis":
        p = os.getenv("ANALYSIS_INPUT_PATH")
        if not p:
            raise EnvironmentError("ANALYSIS_PATH env var not set")
    else:
        raise ValueError(f"unknown filetype: {filetype}")
    return Path(p)


#  Defaults 

def build_defaults(schema: dict) -> dict:
    if schema.get("type") == "object":
        obj = {}
        for name, prop in schema.get("properties", {}).items():
            if "default" in prop:
                obj[name] = prop["default"]
            elif prop.get("type") == "object":
                obj[name] = build_defaults(prop)
            elif prop.get("type") == "array":
                obj[name] = prop.get("default", [])
            elif prop.get("type") == "string":
                obj[name] = prop.get("default", "")
            elif prop.get("type") == "boolean":
                obj[name] = prop.get("default", False)
            elif prop.get("type") in ("integer", "number"):
                obj[name] = prop.get("default", 0)
        return obj
    return schema.get("default", {})


#  API

def new_file(filetype: str, name: str):
    if filetype not in SCHEMAS:
        print(f"unknown filetype '{filetype}'. valid: {', '.join(SCHEMAS)}")
        sys.exit(1)

    folder = get_path(filetype)
    folder.mkdir(parents=True, exist_ok=True)
    filepath = folder / f"{name}.json"

    if filepath.exists():
        print(f"file already exists: {filepath}")
        return

    template = build_defaults(SCHEMAS[filetype])
    with open(filepath, "w") as f:
        json.dump(template, f, indent=2)
    print(f"created {filepath}")


def check_file(filetype: str, name: str) -> bool:
    if filetype not in SCHEMAS:
        print(f"unknown filetype '{filetype}'. valid: {', '.join(SCHEMAS)}")
        sys.exit(1)

    folder = get_path(filetype)
    filepath = folder / f"{name}.json"

    if not filepath.exists():
        print(f"file not found: {filepath}")
        return False

    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"invalid json: {e}")
        return False

    schema = SCHEMAS[filetype]
    validator = Draft7Validator(schema)
    errors = list(validator.iter_errors(data))

    if not errors:
        print(f"valid: {filepath}")
        return True

    for err in errors:
        path = " -> ".join(str(p) for p in err.path) if err.path else "root"
        print(f"  [{path}] {err.message}")
    return False


#  CLI

def main():
    parser = argparse.ArgumentParser(
        description="Create and validate project files")
    subparsers = parser.add_subparsers(dest="action", required=True)

    new_p = subparsers.add_parser("new")
    new_p.add_argument("filetype", choices=list(SCHEMAS))
    new_p.add_argument("name")

    check_p = subparsers.add_parser("check")
    check_p.add_argument("filetype", choices=list(SCHEMAS))
    check_p.add_argument("name")

    args = parser.parse_args()

    if args.action == "new":
        new_file(args.filetype, args.name)
    elif args.action == "check":
        ok = check_file(args.filetype, args.name)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
