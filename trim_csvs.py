#!/usr/bin/env python3
import os
import glob
import pandas as pd
import argparse

def trim_csvs(folder: str, rows: int, output: str | None = None):
    """
    Trims all CSVs in the given folder down to the first `rows` rows.
    
    Args:
        folder (str): Path to the folder containing CSV files.
        rows (int): Number of rows to keep from each CSV.
        output (str | None): Optional output folder. If not provided, files are overwritten.
    """
    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {folder}")
        return
    
    if output:
        os.makedirs(output, exist_ok=True)
    
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            trimmed = df.iloc[:rows]
            
            if output:
                out_path = os.path.join(output, os.path.basename(csv_path))
            else:
                out_path = csv_path  # overwrite original
            
            trimmed.to_csv(out_path, index=False)
            print(f"Trimmed {csv_path} -> {out_path} ({rows} rows)")
        except Exception as e:
            print(f"Failed to process {csv_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim all CSVs in a folder to a fixed number of rows")
    parser.add_argument("folder", help="Path to the folder containing CSV files")
    parser.add_argument("rows", type=int, help="Number of rows to keep")
    parser.add_argument("--output", "-o", help="Optional output folder (default: overwrite in place)")
    
    args = parser.parse_args()
    trim_csvs(args.folder, args.rows, args.output)
