import json
import time
import joblib
import jsonschema
import pandas as pd
import traceback
from pathlib import Path
from typing import Dict, Any, List
import hashlib
import os

from coregtor.forest import create_model, tree_paths
from coregtor.context import create_context, transform_context, compare_context
from coregtor.utils.error import CoRegTorError
from coregtor_cluster import identify_coregulators

from joblib import Parallel, delayed

# keys that affect computation — used for checkpoint hash
COMPUTE_KEYS = ["create_model", "tree_paths", "create_context", "transform_context", "compare_context"]

SCHEMA = {
    "type": "object",
    "required": [],
    "properties": {
        "target_genes": {
            "type": "array",
            "items": {"type": "string"}
        },
        "create_model": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "enum": ["rf", "et"], "default": "rf"},
                "model_options": {"type": "object", "default": {"max_depth": 5, "n_estimators": 1000}}
            }
        },
        "tree_paths": {"type": "object", "default": {}},
        "create_context": {
            "type": "object",
            "properties": {
                "method": {"type": "string", "default": "tree_paths"}
            }
        },
        "transform_context": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "method"],
                "properties": {
                    "id": {"type": "string", "default": "default"},
                    "method": {"type": "string", "default": "gene_frequency"},
                    "normalize": {"type": "boolean", "default": False},
                    "min_frequency": {"type": "integer", "default": 1}
                },
                "additionalProperties": True
            }
        },
        "compare_context": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "method", "transformation_id"],
                "properties": {
                    "id": {"type": "string", "default": "default"},
                    "method": {"type": "string", "default": "cosine"},
                    "transformation_id": {"type": "string", "default": "default"},
                    "convert_to_distance": {"type": "boolean", "default": False}
                },
                "additionalProperties": True
            }
        },
        "checkpointing": {"type": "boolean", "default": True},
        "save_model": {"type": "boolean", "default": False},
        "force_fresh": {"type": "boolean", "default": False},
        "paths": {
            "type": "object",
            "properties": {
                "temp": {"type": "string", "default": ""},
                "output": {"type": "string", "default": ""}
            }
        },
        "clustering": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "matrix_id"],
                "properties": {
                    "id": {"type": "string", "default": "default"},
                    "matrix_id": {"type": "string", "default": "default"},
                    "method": {"type": "string", "enum": ["hierarchical_clustering"], "default": "hierarchical_clustering"},
                    "method_options": {"type": "object", "default": {"auto_threshold": "inconsistency"}}
                },
                "additionalProperties": True
            }
        },
        "result_generation": {
            "type": "object",
            "properties": {
                "n_jobs": {"type": "integer", "default": 1},
                "rerun": {"type": "boolean", "default": False}
            },
            "additionalProperties": True
        }
    }
}


class Pipeline:
    def __init__(self, expression_data: pd.DataFrame, tflist: list,
                 options: Dict[str, Any], exp_title: str = None):

        if expression_data is None:
            raise ValueError("no expression data provided")
        if not tflist:
            raise ValueError("tflist not provided")

        self.expression_data = expression_data
        self.tflist = tflist

        defaults = Pipeline._generate_default_config_dict()
        options = {**defaults, **(options or {})}
        jsonschema.validate(instance=options, schema=SCHEMA)
        self.options = options

        self.results: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {}
        self.status: Dict[str, Any] = {}

        self.title = exp_title.replace(" ", "_") if exp_title else f"exp_{int(time.time())}"

        self.checkpoint_dir = Path(os.path.expandvars(self.options["paths"]["temp"]))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.X_shared = None
        self.feature_columns = None
        self._prepare_shared_X()

    def _get_compute_hash(self) -> str:
        compute_config = {k: self.options[k] for k in COMPUTE_KEYS if k in self.options}
        return hashlib.md5(json.dumps(compute_config, sort_keys=True).encode()).hexdigest()

    def _checkpoint_file(self, target: str) -> Path:
        return self.checkpoint_dir / f"{target}.pkl"

    def _checkpoint_exists(self, target: str) -> bool:
        if not self.options.get("checkpointing", True) or self.options.get("force_fresh", False):
            return False
        f = self._checkpoint_file(target)
        if not f.exists():
            return False
        try:
            checkpoint = joblib.load(f)
            return checkpoint.get("compute_hash") == self._get_compute_hash()
        except:
            return False

    def _save_checkpoint(self, target: str):
        checkpoint = {
            "compute_hash": self._get_compute_hash(),
            "timestamp": time.time(),
            "results": self.results[target],
            "stats": self.stats[target],
            "success": self.status[target]
        }
        joblib.dump(checkpoint, self._checkpoint_file(target), compress=3)
        self.results[target] = None
        self.stats[target] = None
        self.status[target] = None

    def _prepare_shared_X(self):
        self.feature_columns = [g for g in self.tflist if g in self.expression_data.columns]
        if not self.feature_columns:
            raise ValueError("no valid feature columns found")
        self.X_shared = self.expression_data[self.feature_columns]

    def _get_model_input(self, target):
        if target not in self.expression_data.columns:
            raise ValueError(f"target '{target}' not in expression data")
        Y = self.expression_data[[target]]
        if target in self.feature_columns:
            X_cols = [col for col in self.feature_columns if col != target]
            X = self.X_shared[X_cols]
        else:
            X = self.X_shared
        return X, Y

    def run_single_target(self, target: str):
        """Run pipeline for one gene. Raises on failure."""
        if self._checkpoint_exists(target):
            return

        stats = {"timing": {}, "quality": {}}
        results = {}
        status = {"success": False, "error": ""}
        save_model = self.options.get("save_model", False)

        try:
            if target not in self.expression_data.columns:
                raise ValueError(f"target '{target}' not in expression data")

            t = time.perf_counter()
            X, Y = self._get_model_input(target)
            stats["timing"]["model_input"] = time.perf_counter() - t

            t = time.perf_counter()
            model = create_model(X, Y, **self.options.get("create_model"))
            stats["timing"]["model_train"] = time.perf_counter() - t

            if save_model:
                results["model"] = model

            t = time.perf_counter()
            paths = tree_paths(model, X, Y, **self.options.get("tree_paths"))
            stats["timing"]["paths_extract"] = time.perf_counter() - t
            stats["quality"]["n_paths"] = len(paths)
            stats["quality"]["n_unique_roots"] = paths["source"].nunique()
            results["paths"] = paths

            t = time.perf_counter()
            contexts = create_context(paths, **self.options.get("create_context"))
            stats["timing"]["context_create"] = time.perf_counter() - t
            stats["quality"]["n_contexts"] = len(contexts)
            results["contexts"] = contexts

            transform_results = []
            for t_config in self.options.get("transform_context", []):
                t = time.perf_counter()
                transformed = transform_context(contexts, **t_config)
                transform_results.append({"id": t_config["id"], "result": transformed})

            results["transform_results"] = transform_results

            comparison_results = []
            for c_config in self.options.get("compare_context", []):
                tf = next((d for d in transform_results if d["id"] == c_config["transformation_id"]), None)
                if tf is None:
                    continue
                t = time.perf_counter()
                matrix = compare_context(tf["result"], **c_config)
                comparison_results.append({"id": c_config["id"], "result": matrix})

            results["comparison_results"] = comparison_results
            stats["timing"]["total"] = sum(stats["timing"].values())

            status["success"] = True
            self.results[target] = results
            self.stats[target] = stats
            self.status[target] = status

            if self.options.get("checkpointing", True):
                self._save_checkpoint(target)

        except CoRegTorError as e:
            status["success"] = False
            status["error"] = str(e)
            self.results[target] = None
            self.stats[target] = None
            self.status[target] = status
            if self.options.get("checkpointing", True):
                self._save_checkpoint(target)
            raise

    @staticmethod
    def _generate_default_config_dict() -> Dict[str, Any]:
        def extract_defaults(node):
            if node.get("type") == "object" and "properties" in node:
                return {k: extract_defaults(v) for k, v in node["properties"].items()}
            if node.get("type") == "array":
                return node.get("default", [])
            return node.get("default", {
                "string": "", "boolean": False,
                "integer": 0, "number": 0.0,
                "array": [], "object": {}
            }.get(node.get("type"), {}))

        d = extract_defaults(SCHEMA)
        d["target_genes"] = []
        return d


class PipelineResults:
    def __init__(self, options: Dict[str, Any], tflist: list, exp_title: str = None):
        self.tflist = tflist
        jsonschema.validate(instance=options, schema=SCHEMA)
        self.options = options
        self.title = exp_title
        self.checkpoint_dir = Path(os.path.expandvars(self.options["paths"]["temp"]))
        self.output_dir = Path(os.path.expandvars(self.options["paths"]["output"]))

    def get_successful_targets(self) -> List[str]:
        return [f.stem for f in self.checkpoint_dir.glob("*.pkl")]

    def generate_clusters_file(self):
        targets = self.get_successful_targets()
        n_jobs = self.options.get("result_generation", {}).get("n_jobs", 4)
        print(f"processing {len(targets)} targets with {n_jobs} workers")

        all_target_results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(self._process_single_target)(target)
            for target in targets
        )

        method_results = {}
        for target_result in all_target_results:
            if target_result is None:
                continue
            for method_id, df in target_result.items():
                if df is not None:
                    method_results.setdefault(method_id, []).append(df)

        for method_id, dfs in method_results.items():
            if not dfs:
                continue
            combined = pd.concat(dfs, ignore_index=True)
            out = self.output_dir / f"result_{method_id}.csv"
            combined.to_csv(out, index=False)
            print(f"saved {len(combined)} rows to {out}")

    def _process_single_target(self, target: str) -> Dict[str, pd.DataFrame]:
        try:
            checkpoint_file = self.checkpoint_dir / f"{target}.pkl"
            if not checkpoint_file.exists():
                return None

            checkpoint_data = joblib.load(checkpoint_file)
            results_by_method = {}

            for method_config in self.options.get("clustering", []):
                method_id = method_config["id"]
                sim_matrix = self._get_sim_matrix(checkpoint_data, method_config["matrix_id"])
                if sim_matrix is None:
                    continue

                _, clusters_df = identify_coregulators(
                    sim_matrix,
                    target,
                    method=method_config["method"],
                    method_options=method_config["method_options"]
                )

                if clusters_df is None or clusters_df.empty:
                    continue

                clusters_df["method"] = f"{self.title}-coregtor-{method_id}"
                results_by_method[method_id] = clusters_df

            return results_by_method

        except Exception as e:
            traceback.print_exc()
            print(f"error processing {target}: {e}")
            return None

    def _get_sim_matrix(self, checkpoint_data: Dict, matrix_id: str):
        results = checkpoint_data.get("results", {}).get("comparison_results", [])
        found = next((d for d in results if d.get("id") == matrix_id), None)
        return found.get("result") if found else None