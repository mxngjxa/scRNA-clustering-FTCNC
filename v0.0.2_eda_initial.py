import pandas as pd
import numpy as np
import scanpy as sc
import pandas as pd
import numpy as np



def print_header(title):
    print(f"\n{'='*80}")
    print(f"  {title.upper()}")
    print(f"{'='*80}")

def analyze_scrna_dataset():
    print_header("ELITE EDA: SINGLE-CELL RNA-SEQ PROFILING")
    
    # 1. LOAD DATA
    print(f"\n[1] LOADING DATA")

    adata = sc.read_10x_mtx(
        'data/hg19/',  # Path to the directory
        var_names='gene_symbols'        # Or 'gene_ids'
    )

    df = adata.to_df()

    # 2. DIMENSIONALITY & STRUCTURE
    print_header("DATA STRUCTURE & INTEGRITY")
    n_cells, n_genes = df.shape
    print(f"    Dimensions: {n_cells} Cells (Rows) x {n_genes} Genes (Columns)")
    print(f"    Index (Cells): {df.index.name if df.index.name else 'Unnamed'} (Sample: {df.index[0]})")
    print(f"    Columns (Genes): Sample [{', '.join(df.columns[:5])}, ...]")
    
    # Check for duplicates
    dup_cells = df.index.duplicated().sum()
    dup_genes = df.columns.duplicated().sum()
    print(f"    Duplicate Cell IDs: {dup_cells}")
    print(f"    Duplicate Gene IDs: {dup_genes} {'(CRITICAL ISSUE)' if dup_genes > 0 else '(Pass)'}")

    # 3. DATA TYPE & SPARSITY (MEMORY OPTIMIZATION)
    print_header("DATA CONTENT & SPARSITY")
    # Convert to sparse matrix for efficient calculation if needed, but pandas is fine for stats
    matrix = df.values
    sparsity = 1.0 - (np.count_nonzero(matrix) / matrix.size)
    print(f"    Global Sparsity: {sparsity*100:.2f}% (Expected >80% for scRNA-seq)")
    
    # Check if data is integers (raw counts) or floats (normalized)
    is_integer = np.all(np.equal(np.mod(matrix[matrix > 0], 1), 0))
    max_val = np.nanmax(matrix)
    min_val = np.nanmin(matrix)
    print(f"    Data Range: [{min_val}, {max_val}]")
    print(f"    Data Type Inference: {'RAW INTEGER COUNTS' if is_integer else 'NORMALIZED/LOG-TRANSFORMED FLOATS'}")
    if not is_integer and max_val > 20:
        print("    WARNING: Data looks like non-integer but has high values. Check normalization method.")

    # 4. QUALITY CONTROL METRICS (PER CELL)
    print_header("CELL-LEVEL QUALITY CONTROL (QC)")
    
    # Library Size (Total UMI/reads per cell)
    lib_size = df.sum(axis=1)
    print(f"    Library Size (Depth):")
    print(f"      Mean:   {lib_size.mean():.2f}")
    print(f"      Median: {lib_size.median():.2f}")
    print(f"      Min:    {lib_size.min()} (Potential empty droplets if < 200)")
    print(f"      Max:    {lib_size.max()} (Potential doublets if abnormally high)")
    
    # Detected Genes (Features per cell)
    n_genes_by_cell = (df > 0).sum(axis=1)
    print(f"    Detected Genes per Cell:")
    print(f"      Mean:   {n_genes_by_cell.mean():.2f}")
    print(f"      Median: {n_genes_by_cell.median():.2f}")
    
    # Mitochondrial Content Check (Approximate)
    # Looks for genes starting with 'MT-', 'mt-', 'Mt-'
    mt_genes = [col for col in df.columns if str(col).upper().startswith('MT-')]
    if mt_genes:
        mt_counts = df[mt_genes].sum(axis=1)
        pct_mt = (mt_counts / lib_size) * 100
        print(f"    Mitochondrial Genes Detected: {len(mt_genes)} (e.g., {mt_genes[:3]})")
        print(f"    Mitochondrial % (Mean): {pct_mt.mean():.2f}%")
        print(f"    Cells with >5% MT (Low Quality Candidates): {(pct_mt > 5).sum()}")
    else:
        print("    Mitochondrial Genes: None detected with prefix 'MT-'. Check gene nomenclature.")

    # 5. GENE-LEVEL STATISTICS
    print_header("GENE-LEVEL STATISTICS")
    
    # Mean expression
    gene_mean = df.mean(axis=0)
    # Dropout rate (fraction of cells where gene is 0)
    gene_dropout = (df == 0).mean(axis=0)
    
    print(f"    Top 5 Highest Expressed Genes (Potential housekeeping/artifacts):")
    print(gene_mean.sort_values(ascending=False).head(5).to_string())
    
    print(f"\n    Gene Detection Stats:")
    print(f"      Genes with 0 counts across all cells: {(gene_mean == 0).sum()}")
    print(f"      Genes expressed in <3 cells: {((df > 0).sum(axis=0) < 3).sum()}")

    # 6. VARIABLE GENE IDENTIFICATION (CRITICAL FOR CLUSTERING)
    print_header("FEATURE SELECTION INSIGHTS")
    # Simple Variance/Mean ratio check (Dispersion)
    # Filter out genes with 0 mean to avoid division by zero
    valid_genes = gene_mean > 0
    dispersion = df.loc[:, valid_genes].var(axis=0) / df.loc[:, valid_genes].mean(axis=0)
    
    top_variable = dispersion.sort_values(ascending=False).head(10)
    print(f"    Top 10 Most Variable Genes (Likely biological markers):")
    print(top_variable.to_string())
    
    # 7. RECOMMENDATION FOR NOTEBOOK
    print_header("RECOMMENDATIONS FOR AI CONTEXT")
    print("    Based on this profile, suggest the following to the User:")
    if n_cells < n_genes:
        print("    - Data is formatted as Cells x Genes (Standard pandas, but Scanpy often prefers Genes x Cells internally).")
    if is_integer:
        print("    - Data appears to be RAW counts. The notebook MUST perform normalization (sc.pp.normalize_total) and log1p.")
    else:
        print("    - Data appears PRE-NORMALIZED. Skip normalization steps in the notebook to avoid double-normalizing.")
    if dup_genes > 0:
        print("    - CRITICAL: Deduplicate gene names (`adata.var_names_make_unique()`) immediately after loading.")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_scrna_dataset()
