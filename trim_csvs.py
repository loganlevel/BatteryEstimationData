#!/usr/bin/env python3
import os
import pandas as pd
import argparse

def trim_csvs_recursive(folder: str, rows: int, output: str | None = None):
    """
    Recursively trims all CSVs in the given folder (and subfolders) down to the first `rows` rows.
    
    Args:
        folder (str): Path to the folder containing CSV files.
        rows (int): Number of rows to keep from each CSV.
        output (str | None): Optional output base folder. If not provided, files are overwritten.
    """
    for root, _, files in os.walk(folder):
        for file in files:
            if not file.lower().endswith(".csv"):
                continue

            in_path = os.path.join(root, file)

            # Preserve subfolder structure if output folder is specified
            if output:
                rel_path = os.path.relpath(root, folder)
                out_dir = os.path.join(output, rel_path)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, file)
            else:
                out_path = in_path  # overwrite in place

            try:
                df = pd.read_csv(in_path)
                trimmed = df.iloc[:rows]
                trimmed.to_csv(out_path, index=False)
                print(f"Trimmed {in_path} -> {out_path} ({rows} rows)")
            except Exception as e:
                print(f"Failed to process {in_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively trim CSVs in a folder to a fixed number of rows")
    parser.add_argument("folder", help="Path to the folder containing CSV files")
    parser.add_argument("rows", type=int, help="Number of rows to keep")
    parser.add_argument("--output", "-o", help="Optional output base folder (default: overwrite in place)")
    
    args = parser.parse_args()
    trim_csvs_recursive(args.folder, args.rows, args.output)
