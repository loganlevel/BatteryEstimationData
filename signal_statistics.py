#!/usr/bin/env python3
import argparse, os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------
# File discovery & loading
# ---------------------------

def find_csvs(root):
    return [p for p in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)]

def load_csv(path):
    df = pd.read_csv(path)
    # Ensure fault_* columns parse to bool if present
    for c in df.columns:
        if c.startswith("fault_"):
            df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    # 'faults' column treated numeric; missing -> 0
    if "faults" in df.columns:
        df["faults"] = pd.to_numeric(df["faults"], errors="coerce").fillna(0).astype(int)
    return df

# ---------------------------
# SoC mapping & fault detection
# ---------------------------

def first_fault_index(df, ignore_frac=0.10):
    n = len(df)
    if n == 0:
        return None
    start = int(np.floor(ignore_frac * n))
    cond = np.zeros(n, dtype=bool)
    if "faults" in df.columns:
        cond |= (df["faults"].values != 0)
    for c in df.columns:
        if c.startswith("fault_"):
            cond |= df[c].astype(bool).values
    idxs = np.where(cond[start:])[0]
    if len(idxs) == 0:
        return None
    return start + idxs[0]

def soc_vector(length, fault_idx):
    # Map indices [0..fault_idx] to SoC 100..0 linearly
    idx = np.arange(fault_idx + 1, dtype=float)
    return 100.0 * (1.0 - idx / float(fault_idx))

# ---------------------------
# Signal selection helpers
# ---------------------------

def pick_signal_columns(df):
    cols = {}
    cols["relaxed"] = "batt_mV" if "batt_mV" in df.columns else None
    cols["sound"] = ("soundDroopMag_mV" if "soundDroopMag_mV" in df.columns
                     else ("soundDroop_mV" if "soundDroop_mV" in df.columns else None))
    cols["bolt"]  = ("boltDroopMag_mV"  if "boltDroopMag_mV"  in df.columns
                     else ("boltDroop_mV"  if "boltDroop_mV"  in df.columns else None))
    return {k: v for k, v in cols.items() if v is not None}

def per_file_series(df, sig_col, fault_idx):
    x = df[sig_col].values[:fault_idx+1].astype(float)
    soc = soc_vector(len(x), fault_idx)
    return soc, x

# ---------------------------
# Core metric calculations
# ---------------------------

def pooled_slope_local(soc_all, x_all, center_pct, window_pct):
    # Local linear fit slope dx/dSoC in a window around center_pct
    lo = max(0.0, center_pct - window_pct/2.0)
    hi = min(100.0, center_pct + window_pct/2.0)
    m = (soc_all >= lo) & (soc_all <= hi)
    if m.sum() < 3:
        return np.nan
    s = soc_all[m]
    y = x_all[m]
    s1 = s - s.mean()
    denom = np.sum(s1 * s1)
    if denom <= 0:
        return np.nan
    b = np.sum(s1 * (y - y.mean())) / denom  # mV per %SoC
    return b

def sigma_total_at_point(per_trial_series, center_pct, band_pct, adc_noise):
    # Cross-trial spread of mean(x) within a tight SoC band, plus ADC noise in quadrature.
    vals = []
    lo = max(0.0, center_pct - band_pct/2.0)
    hi = min(100.0, center_pct + band_pct/2.0)
    for soc, x in per_trial_series:
        m = (soc >= lo) & (soc <= hi)
        if m.sum() == 0:
            continue
        vals.append(np.nanmean(x[m]))
    if len(vals) < 2:
        return np.nan
    between = np.nanstd(vals, ddof=1)
    return float(np.sqrt(between**2 + adc_noise**2))

def build_pooled_arrays(per_trial_series):
    if not per_trial_series:
        return np.array([]), np.array([])
    soc_all = np.concatenate([s for s, _ in per_trial_series])
    x_all   = np.concatenate([x for _, x in per_trial_series])
    return soc_all, x_all

def safe_metrics(local_slope, sigma_total):
    # Returns (detectability, soc_uncertainty)
    if np.isnan(local_slope) or np.isnan(sigma_total) or local_slope == 0:
        return (np.nan, np.nan)
    D = abs(local_slope) / sigma_total
    U = sigma_total / abs(local_slope)
    return (D, U)

# ---------------------------
# CLI + Orchestration
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute detectability / SoC uncertainty for signals at 10% and 0% SoC.")
    ap.add_argument("root", help="Root folder to search recursively for CSV files")
    ap.add_argument("--adc-noise-mv", type=float, default=0.0,
                    help="ADC/meas noise (mV) added in quadrature to cross-trial spread (default: 0)")
    ap.add_argument("--ignore-startup-frac", type=float, default=0.10,
                    help="Fraction of rows to ignore at start when searching for first fault (default: 0.10)")
    ap.add_argument("--slope-window-pct", type=float, default=10.0,
                    help="SoC window width (%%) for local slope fit (default: 10)")
    ap.add_argument("--sigma-band-pct", type=float, default=2.0,
                    help="SoC band (%%) to compute cross-trial spread at the target SoC (default: 2)")
    ap.add_argument("--min-files", type=int, default=3,
                    help="Minimum CSVs required per signal to report metrics (default: 3)")
    ap.add_argument("--no-show", action="store_true",
                    help="Do not open the matplotlib window")
    args = ap.parse_args()

    paths = find_csvs(args.root)
    results = defaultdict(list)

    # Gather per-file series for each signal
    for p in paths:
        try:
            df = load_csv(p)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
            continue
        if df.empty:
            continue
        fi = first_fault_index(df, ignore_frac=args.ignore_startup_frac)
        if fi is None or fi < 2:
            # No valid fault => cannot map 0% SoC; skip this file
            continue
        sigs = pick_signal_columns(df)
        for name, col in sigs.items():
            try:
                soc, x = per_file_series(df, col, fi)
            except Exception as e:
                print(f"[WARN] Failed parsing series in {p} ({col}): {e}")
                continue
            if len(soc) < 3:
                continue
            results[(name, col)].append((soc, x))

    if not results:
        print("No valid data found.")
        return

    rows_10 = []
    rows_0  = []

    for (name, col), series_list in results.items():
        if len(series_list) < args.min_files:
            continue

        soc_all, x_all = build_pooled_arrays(series_list)

        # Local slope around 10% and near 0% (shift center slightly inside range)
        b10 = pooled_slope_local(soc_all, x_all, center_pct=10.0, window_pct=args.slope_window_pct)
        b0  = pooled_slope_local(soc_all, x_all, center_pct=max(0.0, args.slope_window_pct/2.0), window_pct=args.slope_window_pct)

        # Cross-trial spread at those points
        s10 = sigma_total_at_point(series_list, center_pct=10.0, band_pct=args.sigma_band_pct, adc_noise=args.adc_noise_mv)
        s0  = sigma_total_at_point(series_list, center_pct=max(0.0, args.sigma_band_pct/2.0), band_pct=args.sigma_band_pct, adc_noise=args.adc_noise_mv)

        D10, U10 = safe_metrics(b10, s10)
        D0,  U0  = safe_metrics(b0,  s0)

        rows_10.append({
            "signal": name, "column": col, "N_files": len(series_list),
            "slope_mV_per_%SoC": b10, "sigma_total_mV": s10,
            "detectability": D10, "SoC_uncertainty_%": U10
        })
        rows_0.append({
            "signal": name, "column": col, "N_files": len(series_list),
            "slope_mV_per_%SoC": b0, "sigma_total_mV": s0,
            "detectability": D0, "SoC_uncertainty_%": U0
        })

    df10 = pd.DataFrame(rows_10).sort_values(by="detectability", ascending=False)
    df0  = pd.DataFrame(rows_0 ).sort_values(by="detectability", ascending=False)

    # Print to console
    pd.set_option("display.max_columns", None)
    print("\n=== 10% SoC (ranked by detectability) ===")
    print(df10.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    print("\n=== 0% SoC (ranked by detectability) ===")
    print(df0.to_string(index=False, float_format=lambda v: f"{v:.6g}"))

    # Plot window: bar charts of detectability
    if not args.no_show:
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 1, hspace=0.25)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        fig.suptitle("Detectability by Signal")

        def bar_plot(ax, df, title):
            if df.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                return
            # Order bars by detectability
            labels = [f"{s} ({c})" for s, c in zip(df["signal"], df["column"])]
            vals = df["detectability"].values
            ax.bar(labels, vals)
            ax.set_ylabel("Detectability (|dx/dSoC| / σx)")
            ax.set_title(title)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            # Annotate bars with N_files
            for i, n in enumerate(df["N_files"].values):
                ax.text(i, vals[i], f"N={n}", ha="center", va="bottom", fontsize=8, rotation=0)

        bar_plot(ax1, df10, "At 10% SoC")
        bar_plot(ax2, df0,  "At 0% SoC")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
