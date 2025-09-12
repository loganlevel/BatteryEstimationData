#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute a SINGLE temperature slope K (mV/°C) and a delta threshold D_STRONG (mV)
using only the 10–80% SoC region.

Method:
- For each dataset (root, temp_c), scan CSVs recursively.
- For each CSV:
    * Find 0% SoC as the first fault index, ignoring initial rows (skip startup faults).
    * Map rows to SoC = 100% at start, 0% at first fault.
    * Keep only rows with SoC in [SOC_MIN, SOC_MAX].
    * (Optionally) also filter by an OCV window to reduce SoC-driven variation further.
    * Take the median batt_mV from the remaining rows (per-file representative).
- Per temperature, aggregate per-file medians → one (T, V) point (median across files).
- Fit V = a + K*T (OLS) across temperatures → SINGLE K (mV/°C).
- Using fitted K, build positive residual deltas across temperature pairs:
      r_ij = V_j - (V_i + K*(T_j - T_i))
  Keep r_ij > 0 (false "new battery" direction). Set:
      D_STRONG = percentile(r_ij, DELTA_PERCENTILE) + ADC_MARGIN_MV
- Write per-temp medians plus fitted K and D_STRONG to CSV and print a summary.

Required CSV columns: batt_mV (millivolts), faults (int) and/or fault_* (bool-like).
"""

import os, glob
import numpy as np
import pandas as pd

# =========================
# ========= CONFIG ========
# =========================

DATASETS = [
    {"root": "csvs/manufacturers-cross-temp/temp-m10c", "temp_c": -10.0},
    {"root": "csvs/manufacturers-cross-temp/temp-20C",  "temp_c":  20.0},
    {"root": "csvs/manufacturers-cross-temp/temp-60c",  "temp_c":  58.0},
]

# SoC window to use for estimating K and D_STRONG
SOC_MIN = 10.0   # percent
SOC_MAX = 80.0   # percent

# Fault search / 0% detection
IGNORE_STARTUP_FRAC = 0.10  # ignore first 10% of rows when searching for first fault

# Optional OCV filter to further narrow variation inside the SoC window (set to None to disable)
OCV_FILTER_MV = None         # e.g., (2600.0, 2900.0) or None

# Minimum rows required (after filters) to accept a file's median
MIN_SAMPLES_PER_FILE = 10

# Percentile + margin for D_STRONG
DELTA_PERCENTILE = 99.5      # percentile of positive residuals
ADC_MARGIN_MV    = 10.0      # added margin for ADC/model error

# Output
OUTPUT_CSV = "single_K_and_D_soc10_80.csv"

# =========================
# ========== CODE =========
# =========================

def find_csvs(root: str):
    return [p for p in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)]

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Required signal
    if "batt_mV" not in df.columns:
        raise ValueError("Missing required column 'batt_mV'")
    df["batt_mV"] = pd.to_numeric(df["batt_mV"], errors="coerce")

    # Normalize faults
    if "faults" in df.columns:
        df["faults"] = pd.to_numeric(df["faults"], errors="coerce").fillna(0).astype(int)
    for c in df.columns:
        if str(c).startswith("fault_"):
            df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)

    return df

def first_fault_index(df: pd.DataFrame, ignore_frac: float = 0.10):
    """Return index of first fault after skipping the first ignore_frac of rows."""
    n = len(df)
    if n == 0:
        return None
    start = int(np.floor(ignore_frac * n))
    cond = np.zeros(n, dtype=bool)
    if "faults" in df.columns:
        cond |= (df["faults"].to_numpy() != 0)
    for c in df.columns:
        if str(c).startswith("fault_"):
            cond |= df[c].astype(bool).to_numpy()
    rel = np.where(cond[start:])[0]
    if rel.size == 0:
        return None
    return start + rel[0]

def soc_vector(length: int, fault_idx: int):
    """100% at index 0, 0% at fault_idx (inclusive)."""
    idx = np.arange(fault_idx + 1, dtype=float)
    return 100.0 * (1.0 - idx / float(fault_idx))

def file_median_batt_mv_soc_window(df: pd.DataFrame) -> float | None:
    """Return per-file median batt_mV within the SoC window (and optional OCV window)."""
    fi = first_fault_index(df, ignore_frac=IGNORE_STARTUP_FRAC)
    if fi is None or fi < 2:
        return None

    batt = df["batt_mV"].to_numpy(dtype=float)[:fi+1]
    soc = soc_vector(len(batt), fi)

    # Keep only SOC_MIN..SOC_MAX
    m_soc = (soc >= SOC_MIN) & (soc <= SOC_MAX) & np.isfinite(batt)
    if not np.any(m_soc):
        return None
    v = batt[m_soc]

    # Optional OCV filter inside SoC window
    if OCV_FILTER_MV is not None:
        lo, hi = float(OCV_FILTER_MV[0]), float(OCV_FILTER_MV[1])
        mv = (v >= lo) & (v <= hi)
        v = v[mv]
        if v.size == 0:
            return None

    if v.size < MIN_SAMPLES_PER_FILE:
        return None
    return float(np.median(v))

def fit_line_V_on_T(T, V):
    """Fit V = a + K*T (OLS). Return K, a, R^2, n."""
    T = np.asarray(T, dtype=float)
    V = np.asarray(V, dtype=float)
    m = np.isfinite(T) & np.isfinite(V)
    T, V = T[m], V[m]
    n = T.size
    if n < 2:
        return np.nan, np.nan, np.nan, n
    K, a = np.polyfit(T, V, 1)
    Vhat = K*T + a
    ss_res = np.sum((V - Vhat)**2)
    ss_tot = np.sum((V - np.mean(V))**2)
    R2 = 1.0 - (ss_res/ss_tot if ss_tot > 0 else 0.0)
    return float(K), float(a), float(R2), int(n)

def main():
    # 1) Build per-temp medians (from the 10–80% SoC region)
    per_temp_rows = []
    for ds in DATASETS:
        root = ds["root"]; T = float(ds["temp_c"])
        csvs = find_csvs(root)
        per_file_meds = []
        for p in csvs:
            try:
                df = load_csv(p)
            except Exception as e:
                print(f"[WARN] Skipping {p}: {e}")
                continue
            med = file_median_batt_mv_soc_window(df)
            if med is not None:
                per_file_meds.append(med)
        if per_file_meds:
            V_med = float(np.median(per_file_meds))
            per_temp_rows.append({"temp_c": T, "V_median_mV": V_med, "n_files": len(per_file_meds)})
        else:
            per_temp_rows.append({"temp_c": T, "V_median_mV": np.nan, "n_files": 0})

    pts = pd.DataFrame(per_temp_rows).sort_values("temp_c").reset_index(drop=True)

    # 2) Fit SINGLE K from these SoC-window medians
    K, a, R2, n_temps = fit_line_V_on_T(pts["temp_c"].to_numpy(), pts["V_median_mV"].to_numpy())

    # 3) Build positive residuals across temperature pairs to set D_STRONG
    pos_residuals = []
    Tvals = pts["temp_c"].to_numpy(dtype=float)
    Vvals = pts["V_median_mV"].to_numpy(dtype=float)
    valid = np.isfinite(Tvals) & np.isfinite(Vvals)
    Tvals, Vvals = Tvals[valid], Vvals[valid]

    for i in range(len(Tvals)):
        for j in range(len(Tvals)):
            if i == j:
                continue
            # Predicted V at Tj using point i and slope K
            vij_pred = Vvals[i] + K*(Tvals[j] - Tvals[i])
            rij = Vvals[j] - vij_pred
            if rij > 0:
                pos_residuals.append(rij)

    if pos_residuals:
        D_strong = float(np.percentile(pos_residuals, DELTA_PERCENTILE)) + ADC_MARGIN_MV
    else:
        D_strong = np.nan

    # 4) Write output
    out = pts.copy()
    out["fit_intercept_mV"] = a
    out["fit_K_mV_per_C"] = K
    out["fit_R2"] = R2
    out["fit_n_temps"] = n_temps
    out["D_STRONG_mV"] = D_strong  # repeated for convenience
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote: {OUTPUT_CSV}")
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(out.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    if np.isfinite(K):
        print(f"\nSINGLE K = {K:.3f} mV/°C  (R² = {R2:.3f}, temps used = {n_temps})")
    else:
        print("\nNot enough distinct temperatures to fit K.")
    if np.isfinite(D_strong):
        print(f"D_STRONG  = {D_strong:.1f} mV  (percentile {DELTA_PERCENTILE} + {ADC_MARGIN_MV} mV margin)")
    else:
        print("D_STRONG not computed (insufficient positive residuals).")

if __name__ == "__main__":
    main()
