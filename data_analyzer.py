#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
from typing import Optional, List, Dict

# === CONFIG ===
CSV_DIR = "csvs/all-data-high-st"
TIME_COL = "Time Elapsed (hours)"
TEMP_COL = "temp"
# If your plotting script trims to 160 rows, keep parity here:
ROW_LIMIT = 160

# Any column ending with "_mV" will be treated as a voltage-like series
MV_SUFFIX = "_mV"

# Fault columns (same set used in your plotting script)
FAULT_COLS = [
    "fault_brownout",
    "fault_sound",
    "fault_sound_brownout",
    "fault_bolt",
    "fault_bolt_brownout",
]

OUT_CSV = "low_battery_summary.csv"

def list_csvs(root: str) -> List[str]:
    csvs = []
    for sub in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        csvs.extend(sorted(glob.glob(os.path.join(root, sub, "*.csv"))))
    return csvs

def find_low_battery_index(df: pd.DataFrame) -> Optional[int]:
    """Replicates your logic: find first fault (ignoring first 10% rows),
    then go to time that's 10% BEFORE that, and choose nearest row index."""
    # restrict rows (to mirror plotting)
    if ROW_LIMIT is not None and len(df) > ROW_LIMIT:
        df = df.iloc[:ROW_LIMIT]

    if TIME_COL not in df.columns:
        return None

    fault_cols_present = [c for c in FAULT_COLS if c in df.columns]
    if not fault_cols_present:
        return None

    any_fault = df[fault_cols_present].any(axis=1).copy()
    if not any_fault.any():
        return None

    # Ignore first 10% rows (like your plotting script)
    n_rows = len(df)
    min_idx = int(np.ceil(0.10 * n_rows))
    any_fault.iloc[:min_idx] = False

    if not any_fault.any():
        return None

    idx_first_fault = any_fault.idxmax()

    t = df[TIME_COL]
    t0 = t.iloc[0]
    t_fault = t.loc[idx_first_fault]
    hours_to_fault = float(t_fault - t0)

    # Target time is 10% before the fault time
    t_target = t_fault - 0.10 * hours_to_fault

    # Find nearest time index
    idx_low = (t - t_target).abs().idxmin()
    return idx_low

def mode_temperature(df: pd.DataFrame) -> Optional[float]:
    if TEMP_COL not in df.columns:
        return None
    m = df[TEMP_COL].mode(dropna=True)
    if len(m) == 0:
        return None
    return float(m.iloc[0])

def numeric_derivatives_at_index(df: pd.DataFrame, idx: int, value_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute value, first derivative (dV/dt), and second derivative (d2V/dt2)
    at 'idx' for each column in value_cols.
    Uses numpy.gradient with respect to TIME_COL.
    """
    # Slice to ROW_LIMIT for parity
    if ROW_LIMIT is not None and len(df) > ROW_LIMIT:
        df = df.iloc[:ROW_LIMIT]

    # Ensure numeric and monotonic time (if there is jitter, it still works as gradient uses spacing)
    if TIME_COL not in df.columns:
        return {}
    t = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy()

    # If time has NaNs or is too short, bail gracefully
    if np.isnan(t).any() or len(t) < 3:
        return {col: {"value": np.nan, "d1": np.nan, "d2": np.nan} for col in value_cols}

    results: Dict[str, Dict[str, float]] = {}

    # Build a fast lookup from label index to positional index
    # (since idx could be a label, not positional)
    pos_idx = df.index.get_loc(idx) if not isinstance(df.index.get_loc(idx), slice) else df.index.get_loc(idx).start

    for col in value_cols:
        if col not in df.columns:
            results[col] = {"value": np.nan, "d1": np.nan, "d2": np.nan}
            continue

        y = pd.to_numeric(df[col], errors="coerce").to_numpy()

        # Optionally, fill NaNs to make derivatives stable; keep original for 'value'
        y_filled = pd.Series(y).interpolate(limit_direction="both").bfill().ffill().to_numpy()

        # First derivative dV/dt and second derivative d2V/dt2 with nonuniform spacing
        try:
            d1 = np.gradient(y_filled, t)
            d2 = np.gradient(d1, t)
            val = y[pos_idx]  # original (could be NaN if missing)
            results[col] = {
                "value": float(val) if val == val else np.nan,
                "d1": float(d1[pos_idx]),
                "d2": float(d2[pos_idx]),
            }
        except Exception:
            results[col] = {"value": np.nan, "d1": np.nan, "d2": np.nan}

    return results

def main():
    csv_paths = list_csvs(CSV_DIR)
    if not csv_paths:
        print(f"No CSVs found under {CSV_DIR}")
        return

    rows = []
    printed_header = False

    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if ROW_LIMIT is not None and len(df) > ROW_LIMIT:
            df = df.iloc[:ROW_LIMIT]

        idx_low = find_low_battery_index(df)
        if idx_low is None:
            continue

        # Gather all *_mV columns present (treat as voltage-like)
        mv_cols = [c for c in df.columns if c.endswith(MV_SUFFIX)]

        # Compute derivatives at low-batt index
        derivs = numeric_derivatives_at_index(df, idx_low, mv_cols)

        # Mode temperature
        temp_mode = mode_temperature(df)

        # Build a flat record: subfolder, idx, time, temp_mode, then per-col triples
        subfolder = os.path.basename(os.path.dirname(path))
        t_low = df.loc[idx_low, TIME_COL] if TIME_COL in df.columns else np.nan

        record = {
            "subfolder": subfolder,
            "file": os.path.basename(path),
            "low_batt_idx": idx_low,
            "time_at_low_batt_hours": float(t_low) if t_low == t_low else np.nan,
            "temp_mode_C": temp_mode,
        }

        # Flatten voltage stats
        for col in mv_cols:
            stats = derivs.get(col, {"value": np.nan, "d1": np.nan, "d2": np.nan})
            record[f"{col}__value"] = stats["value"]
            record[f"{col}__d1_per_hour"] = stats["d1"]
            record[f"{col}__d2_per_hour2"] = stats["d2"]

        rows.append(record)

        # Pretty print a concise line for quick inspection
        if not printed_header:
            print("subfolder | file | t(h) | temp_mode(C) | " +
                  " | ".join(f"{c} [V,dV/dt,d2V/dt2]" for c in mv_cols))
            printed_header = True

        triple_strs = []
        for c in mv_cols:
            v = record.get(f"{c}__value", np.nan)
            d1 = record.get(f"{c}__d1_per_hour", np.nan)
            d2 = record.get(f"{c}__d2_per_hour2", np.nan)
            triple_strs.append(f"{c}=({v:.3f},{d1:.3f},{d2:.3f})" if all(np.isfinite([v,d1,d2])) else f"{c}=(nan,nan,nan)")

        print(f"{subfolder} | {os.path.basename(path)} | "
              f"{record['time_at_low_batt_hours']:.3f} | "
              f"{'%.2f'%temp_mode if temp_mode is not None else 'nan'} | "
              + " | ".join(triple_strs))

    if not rows:
        print("No rows to save. Exiting.")
        return

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(out_df)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
