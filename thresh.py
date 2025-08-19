import os
import glob
import pandas as pd
import numpy as np

# === CONFIG ===
CSV_DIR = "csvs/cross_temp"   # parent folder containing subfolders of CSVs
TIME_COL = "Time Elapsed (hours)"
BATT_COL = "batt_mV"

# fault columns to consider (same set as your plotting script)
FAULT_COLS = [
    "fault_brownout",
    "fault_sound",
    "fault_sound_brownout",
    "fault_bolt",
    "fault_bolt_brownout",
]

def first_fault_label(df: pd.DataFrame) -> int | None:
    """
    Returns the index *label* of the first fault, after masking out the first
    10% of rows (to match your plotting heuristic). Returns None if no fault.
    """
    present_fault_cols = [c for c in FAULT_COLS if c in df.columns]
    if not present_fault_cols:
        return None
    any_fault = df[present_fault_cols].any(axis=1)
    if not any_fault.any():
        return None

    # Mask first 10% of rows
    n_rows = len(df)
    min_idx = int(np.ceil(0.10 * n_rows))
    if min_idx > 0:
        any_fault.iloc[:min_idx] = False

    if not any_fault.any():
        return None

    # label of first True
    return any_fault.idxmax()

def find_replacement_index(df: pd.DataFrame) -> int | None:
    """
    Replacement point = 10% of elapsed time before the first fault (with the same 10% row mask).
    Returns the index label of the row whose time is closest to that target.
    """
    if TIME_COL not in df.columns:
        return None

    idx_first_fault = first_fault_label(df)
    if idx_first_fault is None:
        return None

    t_series = df[TIME_COL]
    t0 = t_series.iloc[0]
    t_fault = t_series.loc[idx_first_fault]
    hours_to_fault = t_fault - t0
    t_target = t_fault - 0.10 * hours_to_fault

    return (t_series - t_target).abs().idxmin()

def voltage_at_replacement_point(csv_path: str) -> float | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    # Optionally mirror plotting truncation:
    # df = df.iloc[:150]

    if BATT_COL not in df.columns or TIME_COL not in df.columns:
        return None

    idx = find_replacement_index(df)
    if idx is None:
        return None

    try:
        return float(df.loc[idx, BATT_COL])
    except Exception:
        return None

def dead_voltage_before_first_fault(csv_path: str) -> float | None:
    """
    Dead voltage = battery voltage at the row immediately BEFORE the (masked) first fault.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    # Optionally mirror plotting truncation:
    # df = df.iloc[:150]

    if BATT_COL not in df.columns:
        return None

    idx_fault = first_fault_label(df)
    if idx_fault is None:
        return None

    # Convert label to positional index to get the immediate previous row
    try:
        pos = df.index.get_indexer([idx_fault])[0]
    except Exception:
        return None

    if pos <= 0:
        return None  # no "previous" sample

    prev_label = df.index[pos - 1]
    try:
        return float(df.loc[prev_label, BATT_COL])
    except Exception:
        return None

def main():
    if not os.path.isdir(CSV_DIR):
        print(f"Missing directory: {CSV_DIR}")
        return

    subfolders = sorted(
        d for d in os.listdir(CSV_DIR)
        if os.path.isdir(os.path.join(CSV_DIR, d))
    )
    if not subfolders:
        print(f"No subfolders found in {CSV_DIR}")
        return

    print(f"Scanning {len(subfolders)} subfolders in {CSV_DIR}...\n")

    results = {}  # temp_mode -> (avg_replacement_mV, n_rep, avg_dead_mV, n_dead)

    for sf in subfolders:
        folder_path = os.path.join(CSV_DIR, sf)
        csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        if not csv_files:
            continue

        rep_vals, dead_vals = [], []
        temps = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                if "temp" in df.columns:
                    temps.extend(df["temp"].dropna().tolist())
            except Exception:
                pass

            v_rep = voltage_at_replacement_point(csv_path)
            if v_rep is not None and np.isfinite(v_rep):
                rep_vals.append(v_rep)

            v_dead = dead_voltage_before_first_fault(csv_path)
            if v_dead is not None and np.isfinite(v_dead):
                dead_vals.append(v_dead)

        # Find mode of temperature column for all files in this folder
        temp_mode = None
        if temps:
            temp_counts = pd.Series(temps).mode()
            if not temp_counts.empty:
                temp_mode = temp_counts.iloc[0]

        avg_rep = np.mean(rep_vals) if rep_vals else None
        avg_dead = np.mean(dead_vals) if dead_vals else None
        results[sf] = (temp_mode, avg_rep, len(rep_vals), avg_dead, len(dead_vals))

    # Print nicely
    print("Average voltages by temperature mode:")
    print("-----------------------------------------------------------------------------------")
    print(f"{'Subfolder':>22} | {'Temp Mode':>10} | {'Avg Replacement (mV)':>20} | {'n':>3} | {'Avg Dead (mV)':>14} | {'n':>3}")
    print("-" * 83)
    for sf, (temp_mode, avg_rep, n_rep, avg_dead, n_dead) in results.items():
        rep_str = f"{avg_rep:0.2f}" if avg_rep is not None else "—"
        dead_str = f"{avg_dead:0.2f}" if avg_dead is not None else "—"
        temp_str = f"{temp_mode}" if temp_mode is not None else "—"
        print(f"{sf:>22} | {temp_str:>10} | {rep_str:>20} | {n_rep:>3} | {dead_str:>14} | {n_dead:>3}")
print()

if __name__ == "__main__":
    main()
