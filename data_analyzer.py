#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from scipy.stats import spearmanr

# ========== CONFIG ==========
CSV_DIR   = "csvs/manufacturers-cross-temp"
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
    for dirpath, _, _ in os.walk(root):
        csvs.extend(sorted(glob.glob(os.path.join(dirpath, "*.csv"))))
    return csvs

def device_name_from_basename(basename: str) -> str:
    base = basename.replace(".csv", "")
    parts = base.split("_")
    return parts[2] if len(parts) > 2 else base

def find_low_battery_index(df: pd.DataFrame) -> Optional[int]:
    if ROW_LIMIT is not None and len(df) > ROW_LIMIT:
        df = df.iloc[:ROW_LIMIT]
    if TIME_COL not in df.columns:
        return None

    # Only consider bolt-related faults
    bolt_fault_cols = [c for c in df.columns if c in ["fault_bolt", "fault_bolt_brownout"]]
    if not bolt_fault_cols:
        return None

    any_fault = df[bolt_fault_cols].any(axis=1).copy()
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

def csvs_under_dir(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))

def mode_temperature_over_paths(csv_paths: List[str]) -> Optional[float]:
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

def find_temp_folder(start_path: str) -> Optional[str]:
    cur = os.path.abspath(os.path.dirname(start_path))
    csv_dir_abs = os.path.abspath(CSV_DIR)
    while True:
        base = os.path.basename(cur)
        if base.startswith("temp-"):
            return cur
        parent = os.path.dirname(cur)
        if cur == parent or not cur.startswith(csv_dir_abs):
            return None
        cur = parent

def safe_pos_index(df: pd.DataFrame, label_idx) -> int:
    loc = df.index.get_loc(label_idx)
    if isinstance(loc, slice):
        return loc.start
    return int(loc)

def _series_up_to_idx(df: pd.DataFrame, idx_label, col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return t, value, d1, d2 arrays from start up to and including idx_label.
    Uses interpolation to fill missing values for derivative computation only.
    """
    if TIME_COL not in df.columns or col not in df.columns:
        return np.array([]), np.array([]), np.array([]), np.array([])
    pos = safe_pos_index(df, idx_label)
    t = pd.to_numeric(df[TIME_COL].iloc[:pos+1], errors="coerce").to_numpy()
    y = pd.to_numeric(df[col].iloc[:pos+1], errors="coerce").to_numpy()
    if t.size < 3:
        return t, y, np.array([]), np.array([])
    y_fill = pd.Series(y).interpolate(limit_direction="both").bfill().ffill().to_numpy()
    try:
        d1 = np.gradient(y_fill, t)
        d2 = np.gradient(d1, t)
    except Exception:
        d1 = np.array([])
        d2 = np.array([])
    return t, y, d1, d2

def derivatives_and_noise(df: pd.DataFrame, idx_label, col: str) -> Tuple[float, float, float, float]:
    if ROW_LIMIT is not None and len(df) > ROW_LIMIT:
        df = df.iloc[:ROW_LIMIT]
    if TIME_COL not in df.columns or col not in df.columns:
        return (np.nan, np.nan, np.nan, np.nan)

    t = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy()
    y = pd.to_numeric(df[col], errors="coerce").to_numpy()
    if np.isnan(t).any() or len(t) < 3:
        return (np.nan, np.nan, np.nan, np.nan)

    y_fill = pd.Series(y).interpolate(limit_direction="both").bfill().ffill().to_numpy()
    try:
        d1 = np.gradient(y_fill, t)
        d2 = np.gradient(d1, t)
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan)

    pos = safe_pos_index(df, idx_label)
    v = y[pos] if pos < len(y) else np.nan

    lo = max(0, pos - NOISE_WINDOW)
    hi = min(len(y), pos + NOISE_WINDOW + 1)
    noise = np.nanstd(y[lo:hi]) if hi > lo else np.nan

    return (float(v) if v == v else np.nan,
            float(d1[pos]) if np.isfinite(d1[pos]) else np.nan,
            float(d2[pos]) if np.isfinite(d2[pos]) else np.nan,
            float(noise) if np.isfinite(noise) else np.nan)

def pre_low_batt_monotonicity(df: pd.DataFrame, idx_label, col: str) -> float:
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

def _first_crossing_time(t: np.ndarray, series: np.ndarray, threshold: float) -> Optional[float]:
    """
    Find the first time in t where the series crosses the threshold.
    Direction is chosen from series trend (last vs first).
    Falls back to closest point if no strict crossing exists.
    """
    if t.size == 0 or series.size == 0 or not np.isfinite(threshold):
        return None
    inc = series[-1] >= series[0]
    if inc:
        hits = np.where(series >= threshold)[0]
    else:
        hits = np.where(series <= threshold)[0]
    if hits.size > 0:
        k = int(hits[0])
        if k == 0:
            return float(t[0])
        # linear interpolate around the crossing if possible
        x0, x1 = series[k-1], series[k]
        t0, t1 = t[k-1], t[k]
        if np.isfinite(x0) and np.isfinite(x1) and x1 != x0:
            alpha = (threshold - x0) / (x1 - x0)
            return float(t0 + alpha * (t1 - t0))
        return float(t[k])
    # fallback to nearest point
    k = int(np.nanargmin(np.abs(series - threshold)))
    return float(t[k])

column_name_dict = {
    "batt_mV":"VOC",
    "boltDroopMag_mV":"Bolt Droop",
    "boltDroop_mV":"Bolt Min Voltage",
    "soundDroopMag_mV":"ST Droop",
    "soundDroop_mV":"ST Min Voltage"
}

def summarize_signal_quality(per_file_df: pd.DataFrame,
                             series_cache: Dict[Tuple[str, str], Dict[str, Dict[str, np.ndarray]]]
                             ) -> pd.DataFrame:
    """
    For each *_mV column, compute signal-quality stats (overall and per temp group),
    plus worst-case % timing error if we threshold on the mean metric value.
    """
    records = []

    mv_cols = sorted({c.split("__")[0] for c in per_file_df.columns
                      if c.endswith("__value") and c.endswith("_mV__value")})
    if not mv_cols:
        mv_cols = sorted([c[:-len("__value")] for c in per_file_df.columns
                          if c.endswith("_mV__value")])

    def _error_stats(df_block: pd.DataFrame, scope: str, temp_group: Optional[int]):
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

            N = int(min(len(vals_clean), len(d1_abs), len(noi_clean)))

            mean_val = float(vals_clean.mean()) if not vals_clean.empty else np.nan
            std_val  = float(vals_clean.std(ddof=1)) if len(vals_clean) > 1 else np.nan
            norm_std_val = float(std_val / abs(mean_val)) if (np.isfinite(std_val) and np.isfinite(mean_val) and mean_val != 0) else np.nan

            mean_abs_slope = float(d1_abs.mean()) if not d1_abs.empty else np.nan
            std_abs_slope = float(d1_abs.std(ddof=1)) if len(d1_abs) > 1 else np.nan
            norm_std_abs_slope = float(std_abs_slope / mean_abs_slope) if (np.isfinite(std_abs_slope) and np.isfinite(mean_abs_slope) and mean_abs_slope != 0) else np.nan

            mean_abs_curve = float(d2_abs.mean()) if not d2_abs.empty else np.nan
            std_abs_curve = float(d2_abs.std(ddof=1)) if len(d2_abs) > 1 else np.nan
            norm_std_abs_curve = float(std_abs_curve / mean_abs_curve) if (np.isfinite(std_abs_curve) and np.isfinite(mean_abs_curve) and mean_abs_curve != 0) else np.nan

            mean_noise     = float(noi_clean.mean()) if not noi_clean.empty else np.nan
            mean_abs_rho   = float(rho_abs.mean()) if not rho_abs.empty else np.nan
            snr_slope      = float(mean_abs_slope / mean_noise) if (np.isfinite(mean_abs_slope) and np.isfinite(mean_noise) and mean_noise != 0) else np.nan

            # -------- NEW: worst-case % timing error for thresholds at group means --------
            # thresholds (per group) for each metric:
            thr_value = mean_val
            thr_abs_slope = mean_abs_slope
            thr_abs_curve = mean_abs_curve

            worst_err_value = np.nan
            worst_err_abs_slope = np.nan
            worst_err_abs_curve = np.nan

            # Iterate files in block to evaluate errors using cached series
            errs_value = []
            errs_slope = []
            errs_curve = []

            for _, row in df_block.iterrows():
                key = (row["subfolder"], row["device"])
                cache_for_file = series_cache.get(key, {}).get(col, None)
                if cache_for_file is None:
                    continue
                t = cache_for_file["t"]
                y = cache_for_file["value"]
                s = cache_for_file["abs_slope"]
                c = cache_for_file["abs_curve"]
                t_low = cache_for_file["t_low"]
                if not (np.size(t) and np.isfinite(t_low) and t_low > 0):
                    continue

                # Value metric
                if np.isfinite(thr_value):
                    t_cross = _first_crossing_time(t, y, thr_value)
                    if t_cross is not None and np.isfinite(t_cross):
                        errs_value.append(100.0 * abs(t_cross - t_low) / t_low)

                # |slope| metric
                if np.isfinite(thr_abs_slope) and s.size:
                    t_cross = _first_crossing_time(t, s, thr_abs_slope)
                    if t_cross is not None and np.isfinite(t_cross):
                        errs_slope.append(100.0 * abs(t_cross - t_low) / t_low)

                # |curvature| metric
                if np.isfinite(thr_abs_curve) and c.size:
                    t_cross = _first_crossing_time(t, c, thr_abs_curve)
                    if t_cross is not None and np.isfinite(t_cross):
                        errs_curve.append(100.0 * abs(t_cross - t_low) / t_low)

            if errs_value:
                worst_err_value = float(np.nanmax(errs_value))
            if errs_slope:
                worst_err_abs_slope = float(np.nanmax(errs_slope))
            if errs_curve:
                worst_err_abs_curve = float(np.nanmax(errs_curve))
            # ------------------------------------------------------------------------------

            records.append({
                "scope": scope,
                "temp_group_C": temp_group,
                "column": column_name_dict[col],
                "N": N,
                "mean_noise_std": mean_noise,
                "mean_abs_spearman": mean_abs_rho,
                "SNR_slope": snr_slope,
                "mean_value_at_low": mean_val,
                "std_value_at_low": std_val,
                "norm_std_value_at_low": norm_std_val,
                "worst_pct_error_from_value_threshold": worst_err_value/100.0,
                "mean_abs_slope_at_low": mean_abs_slope,
                "std_abs_slope_at_low": std_abs_slope,
                "norm_std_abs_slope_at_low": norm_std_abs_slope,
                "worst_pct_error_from_abs_slope_threshold": worst_err_abs_slope/100.0,
                "mean_abs_curvature_at_low": mean_abs_curve,
                "std_abs_curvature_at_low": std_abs_curve,
                "norm_std_abs_curvature_at_low": norm_std_abs_curve,
                "worst_pct_error_from_abs_curvature_threshold": worst_err_abs_curve/100.0,
            })

    # Overall
    # _error_stats(per_file_df, scope="overall", temp_group=None)

    # Per temperature group (rounded mode temp)
    if "temp_group_C" in per_file_df.columns:
        for tg, group in per_file_df.groupby("temp_group_C", dropna=True):
            _error_stats(group, scope="per_temp", temp_group=int(tg))

    return pd.DataFrame.from_records(records)

# ========== MAIN PIPELINE ==========

def main():
    paths = list_csvs(CSV_DIR)
    if not paths:
        print(f"No CSVs found under {CSV_DIR}")
        return

    per_file_rows = []
    # in-memory series cache keyed by (subfolder, device) -> per-column metric series up to low-batt
    series_cache: Dict[Tuple[str, str], Dict[str, Dict[str, np.ndarray]]] = {}

    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if ROW_LIMIT is not None and len(df) > ROW_LIMIT:
            df = df.iloc[:ROW_LIMIT]

        if TIME_COL not in df.columns:
            continue

        idx_low = find_low_battery_index(df)
        if idx_low is None:
            continue

        mv_cols = [c for c in df.columns if c.endswith(MV_SUFFIX)]
        if not mv_cols:
            continue

        # ---------- Mode temperature from the temp-XX folder ----------
        temp_folder = find_temp_folder(path)
        if temp_folder is not None:
            csvs_for_temp_folder = csvs_under_dir(temp_folder)
            temp_mode = mode_temperature_over_paths(csvs_for_temp_folder)
        else:
            try:
                temp_mode = float(pd.to_numeric(df[TEMP_COL], errors="coerce").mode(dropna=True).iloc[0])
            except Exception:
                temp_mode = None
        temp_mode_rounded = int(round(temp_mode)) if temp_mode is not None else None
        # ----------------------------------------------------------------

        manufacturer_subfolder = os.path.basename(os.path.dirname(path))
        device = device_name_from_basename(os.path.basename(path))
        key = (manufacturer_subfolder, device)

        rec = {
            "subfolder": manufacturer_subfolder,
            "device": device,
            "temp_mode_C": temp_mode,
            "temp_group_C": temp_mode_rounded,
        }

        t_low = df.loc[idx_low, TIME_COL]
        rec["time_at_low_batt_hours"] = float(t_low) if t_low == t_low else np.nan

        # init per-file cache
        if key not in series_cache:
            series_cache[key] = {}

        for col in mv_cols:
            v, d1, d2, noise = derivatives_and_noise(df, idx_low, col)
            rho = pre_low_batt_monotonicity(df, idx_low, col)

            rec[f"{col}__value"] = v
            rec[f"{col}__d1_per_hour"] = d1
            rec[f"{col}__d2_per_hour2"] = d2
            rec[f"{col}__noise_std"] = noise
            rec[f"{col}__spearman_pre_low"] = rho

            # ---- NEW: cache series up to low-batt for error evaluation ----
            t_ser, y_ser, d1_ser, d2_ser = _series_up_to_idx(df, idx_low, col)
            series_cache[key][col] = {
                "t": t_ser,
                "value": y_ser,
                "abs_slope": np.abs(d1_ser) if d1_ser.size else np.array([]),
                "abs_curve": np.abs(d2_ser) if d2_ser.size else np.array([]),
                "t_low": float(t_low) if t_low == t_low else np.nan,
            }
            # ----------------------------------------------------------------

        per_file_rows.append(rec)

    if not per_file_rows:
        print("No usable rows (no low-battery indices found).")
        return

    per_file_df = pd.DataFrame(per_file_rows)

    # ---- Print concise overall/per-temp summaries (no filenames/paths) ----
    summary_df = summarize_signal_quality(per_file_df, series_cache)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)

    # print("\n=== OVERALL SIGNAL QUALITY (per voltage column) ===")
    # overall = summary_df[summary_df["scope"] == "overall"].drop(columns=["scope", "temp_group_C"])
    # print(overall.sort_values(
    #     ["SNR_slope", "mean_abs_spearman"],
    #     ascending=[False, False]
    # ).to_string(index=False))

    print("\n=== PER-TEMPERATURE SIGNAL QUALITY (per voltage column) ===")
    per_temp = summary_df[summary_df["scope"] == "per_temp"].drop(columns=["scope"])
    per_temp = per_temp.sort_values(
        ["temp_group_C", "SNR_slope", "mean_abs_spearman"],
        ascending=[True, False, False]
    )
    print(per_temp.to_string(index=False))

    # ---- Save CSVs (no filenames/paths in the per-file rows) ----
    per_file_df.to_csv(OUT_PER_FILE, index=False)
    summary_df.drop(columns=["scope"], errors="ignore").to_csv(OUT_SUMMARY, index=False)

    print(f"\nWrote per-file metrics to {OUT_PER_FILE}")
    print(f"Wrote signal-quality summaries to {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
