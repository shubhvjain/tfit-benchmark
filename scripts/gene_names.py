import sqlite3
import os
from pathlib import Path
import pandas as pd

db_path = Path(os.getenv("DATA_PATH"))/"gencode"/"gene_name_mapping.db"

GENCODE_ATTRIBUTES = [
    'gene_id',
    'transcript_id',
    'gene_type',
    'gene_name',
    'transcript_name',
    'protein_id',
    'exon_id'
]

def generate_kv(inp):
    parts = inp.strip().split(' ')
    if len(parts) == 2:
        key = parts[0].strip()
        value = parts[1].replace('"',"").strip()
        return key, value
    return None
    
def process_line(line):
    attribute_str = line.replace('\n',"").split('\t')[-1].replace("'",' ')
    items = attribute_str.split(";")
    result = {}
    for item in items:
        pair = generate_kv(item)
        if pair:
            k, v = pair
            result[k] = v
    return result

def build_database(chunksize=20000):
    if db_path.exists():
        raise ValueError("DB file already exists")
    
    print("Creating database...")
    loc = Path(os.getenv("DATA_PATH"))/"gencode"/"gencode.v39.primary_assembly.annotation.gtf"
    
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA synchronous = OFF")
    con.execute("PRAGMA journal_mode = MEMORY")
    
    first_chunk = True
    
    try:
        with open(loc, "r") as f:
            chunk = []
            for line in f:
                if line.startswith('#'):
                    continue
                
                processed = process_line(line)
                row = {col: processed.get(col, None) for col in GENCODE_ATTRIBUTES}
                chunk.append(row)
                
                if len(chunk) >= chunksize:
                    df = pd.DataFrame(chunk, columns=GENCODE_ATTRIBUTES)
                    df.to_sql(
                        name="mappings",
                        con=con,
                        if_exists='replace' if first_chunk else 'append',
                        index=False
                    )
                    first_chunk = False
                    print(f"Processed {len(chunk)} lines")
                    chunk = []
            
            if chunk:
                df = pd.DataFrame(chunk, columns=GENCODE_ATTRIBUTES)
                df.to_sql(
                    name="mappings",
                    con=con,
                    if_exists='replace' if first_chunk else 'append',
                    index=False
                )
                print(f"Processed final {len(chunk)} lines")
        
        print("SQL file created successfully")
    finally:
        con.execute("CREATE INDEX IF NOT EXISTS idx_gene_id ON mappings(gene_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_gene_name ON mappings(gene_name)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_transcript_id ON mappings(transcript_id)")
        con.close()


def get_mappings(gene_list, source, target):
    """
    Map gene identifiers from source format to target format.
    
    Args:
        gene_list: List of identifiers to map
        source: Source attribute name in database (e.g., 'gene_name', 'gene_id')
        target: Target attribute name in database (e.g., 'gene_id', 'gene_name')
    
    Returns:
        Dictionary mapping {source_value: target_value, ...}
    """
    con = sqlite3.connect(db_path)
    
    try:
        # Remove None/NaN and get unique values
        unique_values = [x for x in set(gene_list) if x is not None and pd.notna(x)]
        
        if len(unique_values) == 0:
            print("No values to map")
            return {}
        
        # Create query
        placeholders = ','.join(['?'] * len(unique_values))
        query = f"""
            SELECT DISTINCT {source}, {target}
            FROM mappings
            WHERE {source} IN ({placeholders})
            AND {target} IS NOT NULL
        """
        
        # Execute query and create mapping dict
        mapping_df = pd.read_sql_query(query, con, params=unique_values)
        mapping_dict = dict(zip(mapping_df[source], mapping_df[target]))
        
        # Report statistics
        total = len(gene_list)
        unique_count = len(unique_values)
        mapped_count = len(mapping_dict)
        failed_count = unique_count - mapped_count
        null_in_results = sum(1 for gene in gene_list if gene not in mapping_dict)
        
        print(f"Mapping from {source} to {target}:")
        print(f"  Total input values: {total}")
        print(f"  Unique values: {unique_count}")
        print(f"  Successfully mapped: {mapped_count}")
        print(f"  Failed to map: {failed_count}")
        print(f"  Null results: {null_in_results} ({null_in_results/total*100:.2f}%)")
        
        return mapping_dict
        
    finally:
        con.close()

if __name__=="__main__":
    build_database(chunksize=200000)

