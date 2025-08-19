import os
import glob
import pandas as pd
import numpy as np

# === CONFIG ===
CSV_DIR = "csvs/cross_temp"   # parent folder containing subfolders of CSVs
TIME_COL = "Time Elapsed (hours)"
BATT_COL = "batt_mV"

# Optional candidate thresholds to evaluate (set to None to skip scoring)
REPLACEMENT_THRESH_MV: float | None = None  # e.g., 2750
DEAD_THRESH_MV: float | None = None         # e.g., 2550

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

def _threshold_cross_label(df: pd.DataFrame, threshold_mv: float) -> int | None:
    """
    Returns the index label of the first row (after masking first 10% rows)
    where batt_mV <= threshold_mv. None if never crossed.
    """
    if BATT_COL not in df.columns or TIME_COL not in df.columns:
        return None

    n_rows = len(df)
    min_idx = int(np.ceil(0.10 * n_rows))
    # Build a boolean Series of crossings
    crossed = df[BATT_COL] <= threshold_mv
    if min_idx > 0:
        crossed.iloc[:min_idx] = False

    if not crossed.any():
        return None

    return crossed.idxmax()

def _file_threshold_metrics(df: pd.DataFrame, threshold_mv: float) -> dict | None:
    """
    For one file: compute timing metrics between threshold-cross time and first fault.
    Returns dict with:
      - dt_hours: positive => threshold earlier than fault (good lead time)
      - dt_frac_of_prefault: dt_hours / (t_fault - t0)
      - crossed_before_fault: bool
    None if no fault or timing columns missing.
    """
    if TIME_COL not in df.columns:
        return None

    idx_fault = first_fault_label(df)
    if idx_fault is None:
        return None

    t = df[TIME_COL]
    t0 = t.iloc[0]
    t_fault = t.loc[idx_fault]
    pre_fault_span = t_fault - t0
    if pre_fault_span <= 0:
        return None

    idx_cross = _threshold_cross_label(df, threshold_mv)
    if idx_cross is None:
        return {
            "dt_hours": np.nan,
            "dt_frac_of_prefault": np.nan,
            "crossed_before_fault": False,
        }

    t_cross = t.loc[idx_cross]
    dt = t_fault - t_cross  # positive if threshold earlier than fault
    crossed_before_fault = (t_cross <= t_fault)

    return {
        "dt_hours": float(dt) if crossed_before_fault else -float(t_cross - t_fault),
        "dt_frac_of_prefault": float(dt / pre_fault_span) if crossed_before_fault else -float((t_cross - t_fault) / pre_fault_span),
        "crossed_before_fault": bool(crossed_before_fault),
    }

def _dispersion_stats(values: list[float]) -> tuple[str, str, str]:
    """
    Return mean, std, and coefficient of variation as strings (or '—' if N/A).
    """
    if not values:
        return "—", "—", "—"
    arr = np.array(values, dtype=float)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
    cv = std / mean if mean else np.nan
    mean_s = f"{mean:.2f}"
    std_s = f"{std:.2f}"
    cv_s = f"{cv:.3f}" if np.isfinite(cv) else "—"
    return mean_s, std_s, cv_s

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

    # results: sf -> (temp_mode, stats dict)
    results = {}

    for sf in subfolders:
        folder_path = os.path.join(CSV_DIR, sf)
        csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        if not csv_files:
            continue

        rep_vals, dead_vals = [], []
        temps = []

        # For threshold scoring
        rep_dt_hours, rep_dt_frac, rep_hit_flags = [], [], []
        dead_dt_hours, dead_dt_frac, dead_hit_flags = [], [], []

        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                if "temp" in df.columns:
                    temps.extend(df["temp"].dropna().tolist())
            except Exception:
                df = None

            # Observed points
            if df is not None:
                idx_rep = find_replacement_index(df) if df is not None else None
                if idx_rep is not None:
                    try:
                        rep_vals.append(float(df.loc[idx_rep, BATT_COL]))
                    except Exception:
                        pass

                v_dead = dead_voltage_before_first_fault(csv_path)
                if v_dead is not None and np.isfinite(v_dead):
                    dead_vals.append(v_dead)

                # Threshold scoring (if configured)
                if REPLACEMENT_THRESH_MV is not None:
                    m = _file_threshold_metrics(df, REPLACEMENT_THRESH_MV)
                    if m is not None:
                        rep_dt_hours.append(m["dt_hours"])
                        rep_dt_frac.append(m["dt_frac_of_prefault"])
                        rep_hit_flags.append(m["crossed_before_fault"])

                if DEAD_THRESH_MV is not None:
                    m = _file_threshold_metrics(df, DEAD_THRESH_MV)
                    if m is not None:
                        dead_dt_hours.append(m["dt_hours"])
                        dead_dt_frac.append(m["dt_frac_of_prefault"])
                        dead_hit_flags.append(m["crossed_before_fault"])

        # Mode of temperature for all files in this folder
        temp_mode = None
        if temps:
            temp_counts = pd.Series(temps).mode()
            if not temp_counts.empty:
                temp_mode = temp_counts.iloc[0]

        # Dispersion/tightness for observed voltages
        rep_mean_s, rep_std_s, rep_cv_s = _dispersion_stats(rep_vals)
        dead_mean_s, dead_std_s, dead_cv_s = _dispersion_stats(dead_vals)

        # Threshold fitness summaries
        rep_metrics = None
        if REPLACEMENT_THRESH_MV is not None and rep_dt_hours:
            dt_hours = np.array([x for x in rep_dt_hours if np.isfinite(x)], dtype=float)
            dt_frac  = np.array([x for x in rep_dt_frac  if np.isfinite(x)], dtype=float)
            fn_rate = 1.0 - (np.sum(rep_hit_flags) / len(rep_hit_flags)) if rep_hit_flags else np.nan
            rep_metrics = {
                "n_scored": len(rep_dt_hours),
                "mean_dt_h": np.nanmean(dt_hours) if dt_hours.size else np.nan,
                "median_dt_h": np.nanmedian(dt_hours) if dt_hours.size else np.nan,
                "mean_dt_frac": np.nanmean(dt_frac) if dt_frac.size else np.nan,
                "median_dt_frac": np.nanmedian(dt_frac) if dt_frac.size else np.nan,
                "fn_rate": fn_rate,
            }

        dead_metrics = None
        if DEAD_THRESH_MV is not None and dead_dt_hours:
            dt_hours = np.array([x for x in dead_dt_hours if np.isfinite(x)], dtype=float)
            dt_frac  = np.array([x for x in dead_dt_frac  if np.isfinite(x)], dtype=float)
            fn_rate = 1.0 - (np.sum(dead_hit_flags) / len(dead_hit_flags)) if dead_hit_flags else np.nan
            dead_metrics = {
                "n_scored": len(dead_dt_hours),
                "mean_dt_h": np.nanmean(dt_hours) if dt_hours.size else np.nan,
                "median_dt_h": np.nanmedian(dt_hours) if dt_hours.size else np.nan,
                "mean_dt_frac": np.nanmean(dt_frac) if dt_frac.size else np.nan,
                "median_dt_frac": np.nanmedian(dt_frac) if dt_frac.size else np.nan,
                "fn_rate": fn_rate,
            }

        results[sf] = {
            "temp_mode": temp_mode,
            "rep_vals": rep_vals,
            "dead_vals": dead_vals,
            "rep_stats": (rep_mean_s, rep_std_s, rep_cv_s, len(rep_vals)),
            "dead_stats": (dead_mean_s, dead_std_s, dead_cv_s, len(dead_vals)),
            "rep_metrics": rep_metrics,
            "dead_metrics": dead_metrics,
        }

    # ------- PRINT -------
    print("Observed voltages by subfolder (tightness of a single threshold):")
    print("---------------------------------------------------------------------------------------------------------------")
    print(f"{'Subfolder':>22} | {'Temp Mode':>9} | {'Rep Mean':>9} | {'Rep Std':>8} | {'Rep CV':>7} | {'n':>3} | "
          f"{'Dead Mean':>10} | {'Dead Std':>8} | {'Dead CV':>7} | {'n':>3}")
    print("-" * 111)
    for sf, info in results.items():
        temp_mode = info["temp_mode"]
        rep_mean_s, rep_std_s, rep_cv_s, n_rep = info["rep_stats"]
        dead_mean_s, dead_std_s, dead_cv_s, n_dead = info["dead_stats"]
        temp_str = f"{temp_mode}" if temp_mode is not None else "—"
        print(f"{sf:>22} | {temp_str:>9} | {rep_mean_s:>9} | {rep_std_s:>8} | {rep_cv_s:>7} | {n_rep:>3} | "
              f"{dead_mean_s:>10} | {dead_std_s:>8} | {dead_cv_s:>7} | {n_dead:>3}")
    print()

    def _print_threshold_block(name: str, thresh_mv: float | None, key: str):
        if thresh_mv is None:
            return
        print(f"Threshold fitness — {name} = {thresh_mv:.1f} mV")
        print("---------------------------------------------------------------------------------------")
        print(f"{'Subfolder':>22} | {'n':>3} | {'Mean Δt (h)':>12} | {'Median Δt (h)':>14} | "
              f"{'Mean Δ% pre':>11} | {'Median Δ% pre':>13} | {'FN rate':>7}")
        print("-" * 87)
        for sf, info in results.items():
            m = info[key]
            if not m:
                print(f"{sf:>22} | {'—':>3} | {'—':>12} | {'—':>14} | {'—':>11} | {'—':>13} | {'—':>7}")
                continue
            n = m["n_scored"]
            mean_dt_h = "—" if np.isnan(m["mean_dt_h"]) else f"{m['mean_dt_h']:.2f}"
            median_dt_h = "—" if np.isnan(m["median_dt_h"]) else f"{m['median_dt_h']:.2f}"
            mean_dt_pct = "—" if np.isnan(m["mean_dt_frac"]) else f"{100*m['mean_dt_frac']:.1f}%"
            median_dt_pct = "—" if np.isnan(m["median_dt_frac"]) else f"{100*m['median_dt_frac']:.1f}%"
            fn_rate = "—" if np.isnan(m["fn_rate"]) else f"{100*m['fn_rate']:.1f}%"
            print(f"{sf:>22} | {n:>3} | {mean_dt_h:>12} | {median_dt_h:>14} | {mean_dt_pct:>11} | {median_dt_pct:>13} | {fn_rate:>7}")
        print()

    _print_threshold_block("REPLACEMENT_THRESH_MV", REPLACEMENT_THRESH_MV, "rep_metrics")
    _print_threshold_block("DEAD_THRESH_MV", DEAD_THRESH_MV, "dead_metrics")

if __name__ == "__main__":
    main()
