#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from scipy.stats import spearmanr

# ========== CONFIG ==========
CSV_DIR   = "csvs/level-li-cross-temp"
TIME_COL  = "Time Elapsed (hours)"
TEMP_COL  = "temp"
ROW_LIMIT = None                 # match your plotting script window
MV_SUFFIX = "_mV"
FAULT_COLS = [
    "fault_brownout",
    "fault_sound",
    "fault_sound_brownout",
    "fault_bolt",
    "fault_bolt_brownout",
]
NOISE_WINDOW = 3                # +/- samples for noise std at low-batt
OUT_PER_FILE = f"{CSV_DIR}/low_batt_per_file.csv"
OUT_SUMMARY  = f"{CSV_DIR}/signal_quality_summary.csv"

# ========== HELPERS ==========

def list_csvs(root: str) -> List[str]:
    csvs = []
    for sub in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        csvs.extend(sorted(glob.glob(os.path.join(root, sub, "*.csv"))))
    return csvs

def device_name_from_basename(basename: str) -> str:
    """
    Your plotting script built legend labels from basename.split('_')[2].
    We'll mirror that to get the device name like 'BE-1'.
    """
    base = basename.replace(".csv", "")
    parts = base.split("_")
    return parts[2] if len(parts) > 2 else base

def find_low_battery_index(df: pd.DataFrame) -> Optional[int]:
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

    n_rows = len(df)
    min_idx = int(np.ceil(0.10 * n_rows))  # ignore first 10% rows
    any_fault.iloc[:min_idx] = False
    if not any_fault.any():
        return None

    idx_first_fault = any_fault.idxmax()
    t = df[TIME_COL]
    t0 = t.iloc[0]
    t_fault = t.loc[idx_first_fault]
    hours_to_fault = float(t_fault - t0)

    # target is 10% BEFORE the fault time
    t_target = t_fault - 0.10 * hours_to_fault
    idx_low = (t - t_target).abs().idxmin()
    return idx_low

def mode_temperature_across_subfolder(csv_paths: List[str]) -> Optional[float]:
    """
    Computes the mode temperature across all rows in all CSVs in a subfolder.
    Returns the most common temperature value (float) or None if not found.
    """
    temps = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            if TEMP_COL in df.columns:
                temps.extend(df[TEMP_COL].dropna().tolist())
        except Exception:
            continue
    if not temps:
        return None
    m = pd.Series(temps).mode(dropna=True)
    if len(m) == 0:
        return None
    return float(m.iloc[0])

def safe_pos_index(df: pd.DataFrame, label_idx) -> int:
    loc = df.index.get_loc(label_idx)
    if isinstance(loc, slice):
        return loc.start
    return int(loc)

def derivatives_and_noise(df: pd.DataFrame, idx_label, col: str) -> Tuple[float, float, float, float]:
    """
    Returns: (value, d1, d2, noise_std) at idx for column 'col'.
    - d1 = dV/dt, d2 = d2V/dt2 (non-uniform dt via numpy.gradient)
    - noise_std computed in a +/- NOISE_WINDOW neighborhood around idx (on original values).
    """
    if ROW_LIMIT is not None and len(df) > ROW_LIMIT:
        df = df.iloc[:ROW_LIMIT]
    if TIME_COL not in df.columns or col not in df.columns:
        return (np.nan, np.nan, np.nan, np.nan)

    t = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy()
    y = pd.to_numeric(df[col], errors="coerce").to_numpy()
    if np.isnan(t).any() or len(t) < 3:
        return (np.nan, np.nan, np.nan, np.nan)

    # interpolate for derivatives (keep original for value/noise)
    y_fill = pd.Series(y).interpolate(limit_direction="both").bfill().ffill().to_numpy()
    try:
        d1 = np.gradient(y_fill, t)
        d2 = np.gradient(d1, t)
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan)

    pos = safe_pos_index(df, idx_label)
    v = y[pos] if pos < len(y) else np.nan

    # noise around idx using original (non-interpolated) values
    lo = max(0, pos - NOISE_WINDOW)
    hi = min(len(y), pos + NOISE_WINDOW + 1)
    noise = np.nanstd(y[lo:hi]) if hi > lo else np.nan

    return (float(v) if v == v else np.nan,
            float(d1[pos]) if np.isfinite(d1[pos]) else np.nan,
            float(d2[pos]) if np.isfinite(d2[pos]) else np.nan,
            float(noise) if np.isfinite(noise) else np.nan)

def pre_low_batt_monotonicity(df: pd.DataFrame, idx_label, col: str) -> float:
    """
    Spearman ρ between time and column from start up to (and including) low-batt index.
    High |ρ| => more monotonic (|ρ| near 1). Returns np.nan if insufficient data.
    """
    if ROW_LIMIT is not None and len(df) > ROW_LIMIT:
        df = df.iloc[:ROW_LIMIT]
    if TIME_COL not in df.columns or col not in df.columns:
        return np.nan
    pos = safe_pos_index(df, idx_label)
    if pos < 3:
        return np.nan
    t = pd.to_numeric(df[TIME_COL].iloc[:pos+1], errors="coerce")
    y = pd.to_numeric(df[col].iloc[:pos+1], errors="coerce")
    if t.isna().any() or y.isna().all():
        return np.nan
    try:
        rho, _ = spearmanr(t, y, nan_policy="omit")
        return float(rho) if np.isfinite(rho) else np.nan
    except Exception:
        return np.nan

def summarize_signal_quality(per_file_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each *_mV column, compute signal-quality stats:
      - N (count of valid rows)
      - mean_value_at_low, std_value_at_low, CV_value_at_low
      - IQR_value_at_low (Q3 - Q1)
      - mean_abs_slope_at_low (|d1|)
      - mean_abs_curvature_at_low (|d2|)
      - mean_noise_std
      - mean_abs_spearman (monotonicity)
      - SNR_slope := mean_abs_slope_at_low / mean_noise_std
    Done both overall and per temperature group (rounded mode temp).
    Returns a tall DF with index [scope, temp_group, column].
    """
    records = []

    # Gather mv columns from the per-file DF
    mv_cols = sorted({c.split("__")[0] for c in per_file_df.columns
                      if c.endswith("__value") and c.endswith("_mV__value")})
    if not mv_cols:
        # fall back to scanning columns for *_mV__value
        mv_cols = sorted([c[:-len("__value")] for c in per_file_df.columns
                          if c.endswith("_mV__value")])

    def calc_block(df_block: pd.DataFrame, scope: str, temp_group: Optional[int]):
        for col in mv_cols:
            vals = df_block[f"{col}__value"]
            d1   = df_block[f"{col}__d1_per_hour"]
            d2   = df_block[f"{col}__d2_per_hour2"]
            noi  = df_block[f"{col}__noise_std"]
            rho  = df_block[f"{col}__spearman_pre_low"]

            vals_clean = vals.replace([np.inf, -np.inf], np.nan).dropna()
            d1_abs = d1.abs().replace([np.inf, -np.inf], np.nan).dropna()
            d2_abs = d2.abs().replace([np.inf, -np.inf], np.nan).dropna()
            noi_clean = noi.replace([np.inf, -np.inf], np.nan).dropna()
            rho_abs = rho.abs().replace([np.inf, -np.inf], np.nan).dropna()

            N = int(min(len(vals_clean), len(d1_abs), len(noi_clean)))  # conservative
            mean_val = float(vals_clean.mean()) if not vals_clean.empty else np.nan
            std_val  = float(vals_clean.std(ddof=1)) if len(vals_clean) > 1 else np.nan
            cv_val   = float(std_val / abs(mean_val)) if (std_val == std_val and mean_val not in [0, np.nan]) else np.nan
            q1 = float(vals_clean.quantile(0.25)) if len(vals_clean) >= 4 else np.nan
            q3 = float(vals_clean.quantile(0.75)) if len(vals_clean) >= 4 else np.nan
            iqr = (q3 - q1) if (q3 == q3 and q1 == q1) else np.nan
            mean_abs_slope = float(d1_abs.mean()) if not d1_abs.empty else np.nan
            std_abs_slope = float(d1_abs.std(ddof=1)) if len(d1_abs) > 1 else np.nan
            mean_abs_curve = float(d2_abs.mean()) if not d2_abs.empty else np.nan
            std_abs_curve = float(d2_abs.std(ddof=1)) if len(d2_abs) > 1 else np.nan
            mean_noise     = float(noi_clean.mean()) if not noi_clean.empty else np.nan
            mean_abs_rho   = float(rho_abs.mean()) if not rho_abs.empty else np.nan
            snr_slope      = float(mean_abs_slope / mean_noise) if (mean_noise and np.isfinite(mean_noise) and mean_noise != 0) else np.nan

            # Normalized std columns
            norm_std_val = float(std_val / abs(mean_val)) if (std_val == std_val and mean_val not in [0, np.nan]) else np.nan
            norm_std_abs_slope = float(std_abs_slope / mean_abs_slope) if (std_abs_slope == std_abs_slope and mean_abs_slope not in [0, np.nan]) else np.nan
            norm_std_abs_curve = float(std_abs_curve / mean_abs_curve) if (std_abs_curve == std_abs_curve and mean_abs_curve not in [0, np.nan]) else np.nan

            records.append({
                "scope": scope,
                "temp_group_C": temp_group,
                "column": col,
                "N": N,
                "mean_value_at_low": mean_val,
                "std_value_at_low": std_val,
                "norm_std_value_at_low": norm_std_val,
                # "CV_value_at_low": cv_val,
                # "IQR_value_at_low": iqr,
                "mean_abs_slope_at_low": mean_abs_slope,
                "std_abs_slope_at_low": std_abs_slope,
                "norm_std_abs_slope_at_low": norm_std_abs_slope,
                "mean_abs_curvature_at_low": mean_abs_curve,
                "std_abs_curvature_at_low": std_abs_curve,
                "norm_std_abs_curvature_at_low": norm_std_abs_curve,
                "mean_noise_std": mean_noise,
                "mean_abs_spearman": mean_abs_rho,
                "SNR_slope": snr_slope,
            })

    # Overall
    calc_block(per_file_df, scope="overall", temp_group=None)

    # Per temperature group (rounded mode temp)
    if "temp_mode_C" in per_file_df.columns:
        per_file_df = per_file_df.copy()
        per_file_df["temp_group_C"] = per_file_df["temp_mode_C"].round().astype("Int64")
        for tg, group in per_file_df.groupby("temp_group_C", dropna=True):
            calc_block(group, scope="per_temp", temp_group=int(tg))

    return pd.DataFrame.from_records(records)

# ========== MAIN PIPELINE ==========

def main():
    paths = list_csvs(CSV_DIR)
    if not paths:
        print(f"No CSVs found under {CSV_DIR}")
        return

    per_file_rows = []

    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if ROW_LIMIT is not None and len(df) > ROW_LIMIT:
            df = df.iloc[:ROW_LIMIT]

        # Skip if required columns are missing
        if TIME_COL not in df.columns:
            continue

        idx_low = find_low_battery_index(df)
        if idx_low is None:
            continue

        # Gather columns to evaluate
        mv_cols = [c for c in df.columns if c.endswith(MV_SUFFIX)]
        if not mv_cols:
            continue

        # Mode temperature (reference & grouping)
        folder_path = os.path.dirname(path)
        csvs_in_folder = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        temp_mode = mode_temperature_across_subfolder(csvs_in_folder)
        temp_mode_rounded = int(round(temp_mode)) if temp_mode is not None else None

        # Identity fields (NO filenames/paths in output/print)
        subfolder = os.path.basename(os.path.dirname(path))
        device = device_name_from_basename(os.path.basename(path))

        # Base record (one row per CSV, with many per-column stats)
        rec = {
            "subfolder": subfolder,
            "device": device,
            "temp_mode_C": temp_mode,
            "temp_group_C": temp_mode_rounded,
        }

        # Time at low-battery
        t_low = df.loc[idx_low, TIME_COL]
        rec["time_at_low_batt_hours"] = float(t_low) if t_low == t_low else np.nan

        # Per-column point metrics
        for col in mv_cols:
            v, d1, d2, noise = derivatives_and_noise(df, idx_low, col)
            rho = pre_low_batt_monotonicity(df, idx_low, col)

            rec[f"{col}__value"] = v
            rec[f"{col}__d1_per_hour"] = d1
            rec[f"{col}__d2_per_hour2"] = d2
            rec[f"{col}__noise_std"] = noise
            rec[f"{col}__spearman_pre_low"] = rho

        per_file_rows.append(rec)

    if not per_file_rows:
        print("No usable rows (no low-battery indices found).")
        return

    per_file_df = pd.DataFrame(per_file_rows)

    # ---- Print concise overall/per-temp summaries (no filenames/paths) ----
    summary_df = summarize_signal_quality(per_file_df)

    # Pretty print: overall first, then per-temp
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 140)

    print("\n=== OVERALL SIGNAL QUALITY (per voltage column) ===")
    overall = summary_df[summary_df["scope"] == "overall"].drop(columns=["scope", "temp_group_C"])
    print(overall.sort_values(["SNR_slope", "mean_abs_spearman"], ascending=[False, False]).to_string(index=False))

    print("\n=== PER-TEMPERATURE SIGNAL QUALITY (per voltage column) ===")
    per_temp = summary_df[summary_df["scope"] == "per_temp"].drop(columns=["scope"])
    # sort by temp group then SNR
    per_temp = per_temp.sort_values(["temp_group_C", "SNR_slope", "mean_abs_spearman"], ascending=[True, False, False])
    print(per_temp.to_string(index=False))

    # ---- Save CSVs (no filenames/paths in the per-file rows) ----
    # Keep only subfolder/device identities (no path), plus metrics
    per_file_df.to_csv(OUT_PER_FILE, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    # Minimal confirmation
    print(f"\nWrote per-file metrics to {OUT_PER_FILE}")
    print(f"Wrote signal-quality summaries to {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
