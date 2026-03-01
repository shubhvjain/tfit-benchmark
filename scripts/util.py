
import pandas as pd
from pathlib import Path
import os
import json
import sqlite3


def get_exp_path() -> Path:
    p = os.getenv("EXP_INPUT_PATH")
    if not p:
        raise EnvironmentError("EXP_PATH not set")
    return Path(p)


def get_temp_path() -> Path:
    p = os.getenv("EXP_TEMP_PATH")
    if not p:
        raise EnvironmentError("EXP_TEMP_PATH not set")
    return Path(p)


def get_output_path() -> Path:
    p = os.getenv("EXP_OUTPUT_PATH")
    if not p:
        raise EnvironmentError("EXP_OUTPUT_PATH not set")
    return Path(p)


def get_data_path() -> Path:
    p = os.getenv("DATA_PATH")
    if not p:
        raise EnvironmentError("DATA_PATH not set")
    return Path(p)


def get_experiment_paths(exp_name, dataset, tool):
    return {
        "exp_file": get_exp_path() / f"{exp_name}.json",
        "temp_folder": get_temp_path() / exp_name / dataset / tool,
        "output_folder": get_output_path() / exp_name / dataset/tool,
        "input_file": get_temp_path() / exp_name / dataset / "input.json",
        "db_file": get_output_path() / exp_name / dataset/tool/"status.db"
    }


def get_mappings(gene_list, source, target, batch_size=900):
    """
    Map gene identifiers from source format to target format.
    Assuming tfitpy setup was run and the source db file is available.
    Args:
        gene_list: List of identifiers to map
        source: Source attribute name in database (e.g., 'gene_name', 'gene_id')
        target: Target attribute name in database (e.g., 'gene_id', 'gene_name')
        batch_size: Number of items to process per query (default 900, under SQLite's 999 limit)
    Returns:
        Dictionary mapping {source_value: target_value, ...}
    """
    db_path = get_data_path()/"gencode"/"gene_name_mapping.db"
    con = sqlite3.connect(db_path)

    try:
        # Remove None/NaN and get unique values
        unique_values = [x for x in set(
            gene_list) if x is not None and pd.notna(x)]
        if len(unique_values) == 0:
            print("No values to map")
            return {}

        # Process in batches to avoid SQLite variable limit
        mapping_dict = {}
        for i in range(0, len(unique_values), batch_size):
            batch = unique_values[i:i + batch_size]
            placeholders = ','.join(['?'] * len(batch))
            query = f"""
                SELECT DISTINCT {source}, {target}
                FROM mappings
                WHERE {source} IN ({placeholders})
                AND {target} IS NOT NULL
            """

            # Execute query and update mapping dict
            mapping_df = pd.read_sql_query(query, con, params=batch)
            mapping_dict.update(
                dict(zip(mapping_df[source], mapping_df[target])))

        # Report statistics
        total = len(gene_list)
        unique_count = len(unique_values)
        mapped_count = len(mapping_dict)
        failed_count = unique_count - mapped_count
        null_in_results = sum(
            1 for gene in gene_list if gene not in mapping_dict)

        print(f"Mapping from {source} to {target}:")
        print(f"  Total input values: {total}")
        print(f"  Unique values: {unique_count}")
        print(f"  Successfully mapped: {mapped_count}")
        print(f"  Failed to map: {failed_count}")
        print(
            f"  Null results: {null_in_results} ({null_in_results/total*100:.2f}%)")

        return mapping_dict

    finally:
        con.close()


def get_dataset(name):
    """Load dataset using metadata 'file_output' field"""
    p = get_data_path() / f"{name}"
    meta_path = p / "metadata.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset '{name}' not found at {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # print("Metadata:", meta)
    file_path = p / meta["file_name"]
    # print("Data file:", file_path)

    # Handle specific file_output types
    output_type = meta.get("file_output", "dataframe").lower()
    read_options = meta.get("read_options", {})

    if output_type == "dataframe":
        # TF list example: single column with custom names
        default_options = {"names": None, "header": 0}
        opts = default_options.copy()
        opts.update(read_options)
        data = pd.read_csv(file_path, **opts)

    elif output_type == "gtex_bulk":
        # Use your custom GCT reader
        data = read_gct(file_path)

    else:
        raise ValueError(f"Unsupported file_output: {output_type}")

    return meta, data


def read_gct(file_path) -> pd.DataFrame:
    """
    Read Gene Cluster Text (GCT) format file into a pandas DataFrame.

    GCT is a tab-delimited format which include
    - Line 1: Version information
    - Line 2: Dimensions (genes x samples)  
    - Line 3+: Header with Name, Description, and sample columns
    - Data rows: Gene information and expression values 

    Assuming there is  "Description" column that has the name of genes have the gene names for each gene row.

    Args:
        file_path (str or Path) : Path to the GCT file.

    Returns:
        pd.DataFrame :  DataFrame with genes as rows and samples as columns. The index is gene_name (from Description column), columns are sample identifiers. Each cell had gene expression levels

    Notes:
    ------
    - Removes the 'Name' column (gene IDs) and uses 'Description' as gene names

    """
    file_path = Path(file_path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Read GCT file, skipping version and dimension lines
        df = pd.read_csv(file_path, skiprows=2, sep="\t")
        # print(df)
        gene = df["Name"].values.tolist()
        # print(gene)
        mps = get_mappings(gene_list=gene, source='gene_id',
                           target='gene_name')
        # print(mps)
        df["Name"] = df["Name"].map(mps)
        if "Name" not in df.columns or "Description" not in df.columns:
            raise ValueError(
                "GCT file must contain 'Name' and 'Description' columns")

        # remove Name column, rename Description to gene_name, set as index
        df = df.drop(columns=["Name"]).rename(
            columns={"Description": "gene_name"})
        df = df.set_index("gene_name")
        df = df.transpose().rename_axis("sample_name")
        return df
    except Exception as e:
        raise ValueError(f"Error reading GCT file {file_path}: {str(e)}")
