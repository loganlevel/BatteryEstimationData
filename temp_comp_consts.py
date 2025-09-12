#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute OCV-banded temperature slopes K (mV/°C) and per-bin Δ-thresholds (mV).

- DATASETS: list of {"root": "...", "temp_c": float}; each root is scanned recursively for CSVs.
- CSVs must contain "batt_mV".
- For each OCV bin:
  * Aggregate per-file medians -> per-temp medians.
  * Fit V = a + K*T to get K (mV/°C).
  * Build positive residuals across temp pairs using K to get D_STRONG_bin as a high percentile.
- Also compute a global D_STRONG across all bins.
"""

import os, glob
import numpy as np
import pandas as pd

# =========================
# ======= CONFIG ==========
# =========================

DATASETS = [
    {"root": "csvs/manufacturers-cross-temp/temp-20cC", "temp_c": 20.0},
    {"root": "csvs/manufacturers-cross-temp/temp-60c",   "temp_c":   58.0},
]

# OCV bin edges (mV). Example: 2400..3000 in 50 mV steps
OCV_BINS_MV = list(range(2400, 3000, 50))

# Robustness knobs
MIN_SAMPLES_PER_FILE_PER_BIN = 5   # per-file minimum samples in a bin
MIN_TEMPS_PER_BIN = 3              # minimum distinct temps needed to fit K
BIN_REP_STAT = "median"            # "median" or "mean" per-file rep

# Δ-threshold settings
DELTA_PERCENTILE = 99.5            # percentile of positive residuals
ADC_MARGIN_MV = 10.0               # added to threshold (covers ADC/small modeling error)

# Output
OUTPUT_CSV = "ocv_banded_K.csv"

# =========================
# =======  CODE  ==========
# =========================

def find_csvs(root):
    return [p for p in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)]

def load_csv(path):
    df = pd.read_csv(path)
    if "batt_mV" not in df.columns:
        raise ValueError("Missing required column 'batt_mV'")
    df["batt_mV"] = pd.to_numeric(df["batt_mV"], errors="coerce")
    return df

def per_file_rep(vals):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0: return None
    return float(np.median(v) if BIN_REP_STAT == "median" else np.mean(v))

def fit_line_V_on_T(T, V):
    T = np.asarray(T, float); V = np.asarray(V, float)
    m = (~np.isnan(T)) & (~np.isnan(V))
    T = T[m]; V = V[m]
    if T.size < 2:
        return (np.nan, np.nan, np.nan, 0)
    K, intercept = np.polyfit(T, V, 1)
    Vhat = K*T + intercept
    ss_res = np.sum((V - Vhat)**2)
    ss_tot = np.sum((V - np.mean(V))**2)
    R2 = 1.0 - (ss_res/ss_tot if ss_tot > 0 else 0.0)
    return float(K), float(intercept), float(R2), int(T.size)

def main():
    # Build bins
    edges = np.array(OCV_BINS_MV, float)
    if edges.size < 2: raise ValueError("Need at least two OCV bin edges.")
    bins = list(zip(edges[:-1], edges[1:]))

    # For each dataset(temp), collect per-file reps per bin
    pertemp_bin_file_reps = {}  # temp_c -> list of [list of per-file reps] per bin
    pertemp_file_counts = {}    # temp_c -> number of files seen

    for ds in DATASETS:
        root, T = ds["root"], float(ds["temp_c"])
        bin_lists = [[] for _ in bins]
        file_count = 0

        for pth in find_csvs(root):
            try:
                df = load_csv(pth)
            except Exception as e:
                print(f"[WARN] Skipping {pth}: {e}")
                continue
            batt = df["batt_mV"].to_numpy(dtype=float)
            if batt.size == 0: continue
            file_count += 1

            for bidx, (lo, hi) in enumerate(bins):
                m = (batt >= lo) & (batt < hi)
                if np.count_nonzero(m) >= MIN_SAMPLES_PER_FILE_PER_BIN:
                    rep = per_file_rep(batt[m])
                    if rep is not None:
                        bin_lists[bidx].append(rep)

        pertemp_bin_file_reps[T] = bin_lists
        pertemp_file_counts[T] = file_count

    # Collapse per-file reps -> per-temp medians per bin
    pertemp_bin_medians = {}  # temp_c -> list of medians per bin (or None)
    for T, lists in pertemp_bin_file_reps.items():
        medians = []
        for reps in lists:
            if len(reps) == 0:
                medians.append(None)
            else:
                medians.append(float(np.median(reps)))
        pertemp_bin_medians[T] = medians

    # Fit K per bin and collect residuals for Δ-threshold
    rows = []
    all_pos_residuals = []

    for bidx, (lo, hi) in enumerate(bins):
        Ts, Vs = [], []
        for T, med_list in pertemp_bin_medians.items():
            v = med_list[bidx]
            if v is not None and not np.isnan(v):
                Ts.append(float(T)); Vs.append(float(v))

        if len(Ts) >= MIN_TEMPS_PER_BIN:
            K, B, R2, n_used = fit_line_V_on_T(Ts, Vs)
        else:
            K, B, R2, n_used = (np.nan, np.nan, np.nan, len(Ts))

        # Positive residuals to set D_STRONG for this bin
        pos_res = []
        if not np.isnan(K):
            # For each pair of temps (i->j), compute r_ij = Vj - (Vi + K*(Tj-Ti))
            for i in range(len(Ts)):
                for j in range(len(Ts)):
                    if i == j: continue
                    rij = Vs[j] - (Vs[i] + K*(Ts[j] - Ts[i]))
                    if rij > 0:
                        pos_res.append(rij)
                        all_pos_residuals.append(rij)

        if pos_res:
            D_strong_bin = float(np.percentile(pos_res, DELTA_PERCENTILE)) + ADC_MARGIN_MV
        else:
            D_strong_bin = np.nan

        rows.append({
            "bin_lo_mV": lo,
            "bin_hi_mV": hi,
            "bin_center_mV": 0.5*(lo+hi),
            "K_mV_per_C": K,
            "intercept_mV": B,
            "R2": R2,
            "n_temps": n_used,
            "D_STRONG_mV": D_strong_bin,
            "pos_residuals_count": len(pos_res)
        })

    # Global Δ-threshold (fallback)
    if all_pos_residuals:
        D_global = float(np.percentile(all_pos_residuals, DELTA_PERCENTILE)) + ADC_MARGIN_MV
    else:
        D_global = np.nan

    out = pd.DataFrame(rows).sort_values("bin_lo_mV").reset_index(drop=True)
    out["D_STRONG_global_mV"] = D_global
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote: {OUTPUT_CSV}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    if not np.isnan(D_global):
        print(f"\nGlobal D_STRONG ≈ {D_global:.1f} mV (percentile {DELTA_PERCENTILE} + {ADC_MARGIN_MV} mV margin)")
    else:
        print("\nGlobal D_STRONG not computed (insufficient residuals).")

if __name__ == "__main__":
    main()
