"""
Convert raw single-cell data to h5ad format.

Supported input formats
-----------------------
10x MTX   : folder containing matrix.mtx(.gz), barcodes.tsv(.gz), features/genes.tsv(.gz)
10x H5    : .h5 file produced by Cell Ranger (filtered_feature_bc_matrix.h5 etc.)
CSV/TSV   : dense counts matrix, cells × genes or genes × cells
Loom      : .loom file

Usage
-----
# 10x MTX folder
python scripts/convert_to_h5ad.py --input /data/pbmc3k/hg19 --output data/pbmc3k/pbmc3k_raw.h5ad

# 10x H5 file
python scripts/convert_to_h5ad.py --input /data/sample/filtered_feature_bc_matrix.h5 --output data/sample/sample_raw.h5ad

# Dense CSV (genes as rows, cells as columns)
python scripts/convert_to_h5ad.py --input /data/sample/counts.csv --format csv --transpose --output data/sample/sample_raw.h5ad

# Loom
python scripts/convert_to_h5ad.py --input /data/sample/sample.loom --output data/sample/sample_raw.h5ad
"""

import argparse
import os
import sys

import scanpy as sc
import anndata as ad


def detect_format(input_path: str) -> str:
    """Infer format from path if not specified explicitly."""
    if os.path.isdir(input_path):
        return "10x_mtx"
    ext = os.path.splitext(input_path)[1].lower()
    if ext in (".h5",):
        return "10x_h5"
    if ext in (".loom",):
        return "loom"
    if ext in (".csv",):
        return "csv"
    if ext in (".tsv", ".txt"):
        return "tsv"
    sys.exit(
        f"ERROR: Cannot infer format from '{input_path}'. "
        "Pass --format explicitly (10x_mtx | 10x_h5 | csv | tsv | loom)."
    )


def load_data(input_path: str, fmt: str, var_names: str, transpose: bool) -> ad.AnnData:
    if fmt == "10x_mtx":
        print(f"  Reading 10x MTX directory: {input_path}")
        adata = sc.read_10x_mtx(input_path, var_names=var_names, cache=False)

    elif fmt == "10x_h5":
        print(f"  Reading 10x H5 file: {input_path}")
        adata = sc.read_10x_h5(input_path)

    elif fmt in ("csv", "tsv"):
        sep = "," if fmt == "csv" else "\t"
        print(f"  Reading {'CSV' if fmt == 'csv' else 'TSV'} file: {input_path}")
        adata = sc.read_csv(input_path, delimiter=sep)
        if transpose:
            print("  Transposing so rows=cells, columns=genes ...")
            adata = adata.T

    elif fmt == "loom":
        print(f"  Reading Loom file: {input_path}")
        adata = sc.read_loom(input_path)

    else:
        sys.exit(f"ERROR: Unknown format '{fmt}'.")

    return adata


def prompt_interactive() -> argparse.Namespace:
    """Interactively prompt the user for all required inputs."""
    print("\n=== Convert to h5ad ===\n")

    while True:
        input_path = input("Input path (file or folder): ").strip().strip("'\"")
        abs_path = os.path.abspath(os.path.expanduser(input_path))
        if os.path.exists(abs_path):
            input_path = abs_path
            break
        print(f"  Path not found: '{abs_path}'. Please try again.")

    fmt = detect_format(input_path)
    print(f"  Detected format: {fmt}")

    transpose = False
    if fmt in ("csv", "tsv"):
        resp = input("  Are genes rows and cells columns? Transpose? [y/N]: ").strip().lower()
        transpose = resp == "y"

    default_out = os.path.splitext(os.path.basename(input_path.rstrip("/")))[0] + "_raw.h5ad"
    output_path = input(f"Output .h5ad path [{default_out}]: ").strip()
    if not output_path:
        output_path = default_out

    ns = argparse.Namespace(
        input=input_path,
        output=output_path,
        format=fmt,
        var_names="gene_symbols",
        transpose=transpose,
    )
    return ns


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw scRNA-seq data to AnnData h5ad format"
    )
    parser.add_argument(
        "--input",
        help="Path to input data (directory for 10x MTX, file otherwise)"
    )
    parser.add_argument(
        "--output",
        help="Path for the output .h5ad file"
    )
    parser.add_argument(
        "--format",
        choices=["10x_mtx", "10x_h5", "csv", "tsv", "loom"],
        default=None,
        help="Input format. Auto-detected from path if omitted."
    )
    parser.add_argument(
        "--var_names",
        choices=["gene_symbols", "gene_ids"],
        default="gene_symbols",
        help="Which field to use as var (gene) index for 10x data (default: gene_symbols)"
    )
    parser.add_argument(
        "--transpose",
        action="store_true",
        help="Transpose CSV/TSV input (use when rows=genes, columns=cells)"
    )
    args = parser.parse_args()

    # If no arguments provided, fall back to interactive prompts
    if args.input is None:
        args = prompt_interactive()
    elif not os.path.exists(args.input):
        sys.exit(f"ERROR: Input path not found: {args.input}")

    fmt = args.format or detect_format(args.input)
    print(f"\nFormat  : {fmt}")
    print(f"Input   : {args.input}")
    print(f"Output  : {args.output}\n")

    adata = load_data(args.input, fmt, args.var_names, args.transpose)
    adata.var_names_make_unique()

    print(f"\n  Cells : {adata.n_obs:,}")
    print(f"  Genes : {adata.n_vars:,}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    adata.write_h5ad(args.output)
    print(f"\n  Saved → {args.output}\n")


if __name__ == "__main__":
    main()
