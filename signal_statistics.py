#!/usr/bin/env python3

import argparse, os, glob
import pandas as pd
import numpy as np
from collections import defaultdict


# =========================
# =======  CONFIG  ========
# =========================

# Target SoC points to evaluate (percent). Example: [10.0, 0.0] or [15.0, 5.0]
ESTIMATED_LIFE_DAYS = {
    "Lock Pro": 207,
    "Lock Plus": 139,
    "Bolt": 323
}
DAYS_LEFT_TARGETS = {
    "Lock Pro": [25, 4],
    "Lock Plus": [25, 4],
    "Bolt": [25, 4]
}

# Compute the list of target SoC points (flatten and deduplicate)
TARGET_SOC_POINTS = sorted(set(
    float(np.ceil(100 * (days_left / ESTIMATED_LIFE_DAYS[product])))
    for product, targets in DAYS_LEFT_TARGETS.items()
    for days_left in targets
))

# Local slope & sigma windows (in %SoC)
SLOPE_WINDOW_PCT = 10.0
SIGMA_BAND_PCT   = 2.0

# 0% handling convention (for stability/auditability)
# "edge" -> center windows at 0% (slope over [0, W/2], sigma over [0, B/2])
# "mid"  -> center windows at half-width (old behavior; slope over [0, W], sigma over [0, B])
ZERO_SLOPE_CENTER = "mid"   # "edge" or "mid"
ZERO_SIGMA_CENTER = "mid"   # "edge" or "mid"

# Fault search / noise / dataset requirements
IGNORE_STARTUP_FRAC = 0.10  # ignore first 10% rows when finding first fault
ADC_NOISE_MV        = 0.0   # meas noise added in quadrature to cross-file spread
MIN_FILES           = 3     # min CSVs per signal

# Neyman–Pearson thresholding
NP_ALPHA_MISS       = 0.01  # cap missed-low (false-late) at 1%
NP_POS_BAND_BELOW   = 2.0   # pos class: [max(0, p - this), p]
NP_NEG_GAP          = 5.0   # exclude [p, p+gap]; negatives start after gap
NP_NEG_BAND_WIDTH   = 10.0  # neg class: [p+gap, p+gap+width]

# Cross-validation
DO_LOOCV            = True  # if False, falls back to pooled

# Output CSV filename
OUTPUT_CSV          = "detectability_thresholds_min.csv"

# =========================
# =======  CODE  ==========
# =========================

def find_csvs(root):
    return [p for p in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)]

def load_csv(path):
    df = pd.read_csv(path)
    if "faults" in df.columns:
        df["faults"] = pd.to_numeric(df["faults"], errors="coerce").fillna(0).astype(int)
    for c in df.columns:
        if c.startswith("fault_"):
            df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    return df

def first_fault_index(df, ignore_frac=0.10):
    n = len(df)
    if n == 0: return None
    start = int(np.floor(ignore_frac * n))
    cond = np.zeros(n, dtype=bool)
    if "faults" in df.columns:
        cond |= (df["faults"].values != 0)
    for c in df.columns:
        if c.startswith("fault_"):
            cond |= df[c].astype(bool).values
    rel = np.where(cond[start:])[0]
    return (start + rel[0]) if len(rel) else None

def soc_vector(length, fault_idx):
    idx = np.arange(fault_idx + 1, dtype=float)
    return 100.0 * (1.0 - idx / float(fault_idx))

def pick_signal_columns(df):
    cols = {}
    cols["relaxed"] = "batt_mV" if "batt_mV" in df.columns else None
    cols["sound"]   = ("soundDroopMag_mV" if "soundDroopMag_mV" in df.columns
                       else ("soundDroop_mV" if "soundDroop_mV" in df.columns else None))
    cols["bolt"]    = ("boltDroopMag_mV"  if "boltDroopMag_mV"  in df.columns
                       else ("boltDroop_mV"  if "boltDroop_mV"  in df.columns else None))
    return {k: v for k, v in cols.items() if v is not None}

def per_file_series(df, sig_col, fault_idx):
    x = df[sig_col].values[:fault_idx+1].astype(float)
    soc = soc_vector(len(x), fault_idx)
    return soc, x

def pooled_slope_local(soc_all, x_all, center_pct, window_pct):
    lo = max(0.0, center_pct - window_pct/2.0)
    hi = min(100.0, center_pct + window_pct/2.0)
    m = (soc_all >= lo) & (soc_all <= hi)
    if m.sum() < 3: return np.nan
    s = soc_all[m]; y = x_all[m]
    s1 = s - s.mean()
    denom = np.sum(s1*s1)
    if denom <= 0: return np.nan
    return np.sum(s1*(y - y.mean())) / denom  # mV / %SoC

def sigma_total_at_point(per_trial_series, center_pct, band_pct, adc_noise):
    lo = max(0.0, center_pct - band_pct/2.0)
    hi = min(100.0, center_pct + band_pct/2.0)
    vals = []
    for soc, x in per_trial_series:
        m = (soc >= lo) & (soc <= hi)
        if m.sum() == 0: continue
        vals.append(float(np.nanmean(x[m])))
    if len(vals) < 2: return np.nan
    between = np.nanstd(vals, ddof=1)
    return float(np.sqrt(between**2 + adc_noise**2))

def build_pooled_arrays(per_trial_series):
    if not per_trial_series: return np.array([]), np.array([])
    soc_all = np.concatenate([s for s,_ in per_trial_series])
    x_all   = np.concatenate([x for _,x in per_trial_series])
    return soc_all, x_all

def class_bands(soc, p, pos_below, neg_gap, neg_width):
    pos_lo = max(0.0, p - pos_below); pos_hi = p
    neg_lo = min(100.0, p + neg_gap); neg_hi = min(100.0, p + neg_width + neg_gap)
    m_pos = (soc >= pos_lo) & (soc <= pos_hi)
    m_neg = (soc >= neg_lo) & (soc <= neg_hi)
    return m_pos, m_neg

def summarize_per_file_for_np(series_list, p, pos_below, neg_gap, neg_width):
    x_pos, x_neg = [], []
    for soc, x in series_list:
        m_pos, m_neg = class_bands(soc, p, pos_below, neg_gap, neg_width)
        if m_pos.any(): x_pos.append(float(np.nanmean(x[m_pos])))
        if m_neg.any(): x_neg.append(float(np.nanmean(x[m_neg])))
    return np.array(x_pos), np.array(x_neg)

def direction_from_slope(b_local):
    if np.isnan(b_local) or abs(b_local) < 1e-12:
        return None  # unknown, decide from class medians later
    return (">=") if (b_local < 0) else ("<=")

def direction_from_classes(x_pos, x_neg):
    if x_pos.size == 0 or x_neg.size == 0: return "<="
    mpos = np.median(x_pos); mneg = np.median(x_neg)
    # If low-SoC values are larger than high-SoC values -> ">="
    return (">=") if (mpos > mneg) else ("<=")

def sweep_np_threshold(x_pos, x_neg, direction, alpha):
    if x_pos.size == 0 or x_neg.size == 0:
        return (np.nan, np.nan, np.nan)
    xs = np.unique(np.concatenate([x_pos, x_neg]))
    eps = 1e-9 * (xs[-1] - xs[0] if xs.size > 1 and xs[-1] != xs[0] else 1.0)

    candidates = []
    for theta in xs:
        if direction == ">=":
            tpr = np.mean(x_pos >= theta); fpr = np.mean(x_neg >= theta)
            tpr2 = np.mean(x_pos >= (theta - eps)); fpr2 = np.mean(x_neg >= (theta - eps))
        else:
            tpr = np.mean(x_pos <= theta); fpr = np.mean(x_neg <= theta)
            tpr2 = np.mean(x_pos <= (theta + eps)); fpr2 = np.mean(x_neg <= (theta + eps))
        candidates.append((theta, tpr, fpr))
        candidates.append(((theta - eps) if direction==">=" else (theta + eps), tpr2, fpr2))

    feasible = [(th,tpr,fpr) for th,tpr,fpr in candidates if tpr >= (1.0 - alpha)]
    if feasible:
        feasible.sort(key=lambda z: (z[2], -z[1]))
        return feasible[0]
    # fallback: max TPR, then min FPR
    max_tpr = max(z[1] for z in candidates)
    near = [z for z in candidates if abs(z[1]-max_tpr) < 1e-12]
    near.sort(key=lambda z: z[2])
    return near[0]

def loocv_np_threshold(series_list, p, direction, alpha, pos_below, neg_gap, neg_width):
    if len(series_list) < 2:
        return (np.nan, np.nan, np.nan, 0)

    thetas, tprs, fprs = [], [], []
    for i in range(len(series_list)):
        train = series_list[:i] + series_list[i+1:]
        test  = [series_list[i]]
        x_pos_tr, x_neg_tr = summarize_per_file_for_np(train, p, pos_below, neg_gap, neg_width)
        theta, _, _ = sweep_np_threshold(x_pos_tr, x_neg_tr, direction, alpha)
        x_pos_te, x_neg_te = summarize_per_file_for_np(test, p, pos_below, neg_gap, neg_width)
        if np.isnan(theta) or x_pos_te.size==0 or x_neg_te.size==0: continue
        if direction == ">=":
            tprs.append(float(np.mean(x_pos_te >= theta))); fprs.append(float(np.mean(x_neg_te >= theta)))
        else:
            tprs.append(float(np.mean(x_pos_te <= theta))); fprs.append(float(np.mean(x_neg_te <= theta)))
        thetas.append(theta)

    if not thetas: return (np.nan, np.nan, np.nan, 0)
    return (float(np.median(thetas)), float(np.mean(tprs)), float(np.mean(fprs)), len(thetas))

def center_for_zero(p, width, mode):
    if p > 0: return p
    return 0.0 if mode == "edge" else max(0.0, width/2.0)

def main():
    parser = argparse.ArgumentParser(description="Compute minimal detectability & thresholds for battery signals.")
    parser.add_argument("root", help="Root folder (searched recursively) for CSV files.")
    parser.add_argument("--out", default=OUTPUT_CSV, help=f"Output CSV path (default: {OUTPUT_CSV})")
    args = parser.parse_args()

    paths = find_csvs(args.root)
    if not paths:
        print("No CSV files found."); return

    series_by_signal = defaultdict(list)
    for pth in paths:
        try:
            df = load_csv(pth)
        except Exception as e:
            print(f"[WARN] Could not read {pth}: {e}"); continue
        if df.empty: continue
        fi = first_fault_index(df, ignore_frac=IGNORE_STARTUP_FRAC)
        if fi is None or fi < 2: continue
        for name, col in pick_signal_columns(df).items():
            try:
                soc, x = per_file_series(df, col, fi)
                if len(soc) >= 3: series_by_signal[(name, col)].append((soc, x))
            except Exception as e:
                print(f"[WARN] Failed parsing series in {pth} ({col}): {e}")

    rows = []
    for (name, col), series_list in series_by_signal.items():
        if len(series_list) < MIN_FILES: continue
        soc_all, x_all = build_pooled_arrays(series_list)

        for p in TARGET_SOC_POINTS:
            # Centers chosen per convention
            slope_center = center_for_zero(p, SLOPE_WINDOW_PCT, ZERO_SLOPE_CENTER)
            sigma_center = center_for_zero(p, SIGMA_BAND_PCT, ZERO_SIGMA_CENTER)

            # Detectability / Uncertainty
            b_local  = pooled_slope_local(soc_all, x_all, center_pct=slope_center, window_pct=SLOPE_WINDOW_PCT)
            sigma_tot = sigma_total_at_point(series_list, center_pct=sigma_center, band_pct=SIGMA_BAND_PCT, adc_noise=ADC_NOISE_MV)
            if np.isnan(b_local) or np.isnan(sigma_tot) or abs(b_local) < 1e-12:
                detect = np.nan; soc_unc = np.nan
            else:
                detect = abs(b_local) / sigma_tot
                soc_unc = sigma_tot / abs(b_local)

            # Direction: prefer slope sign; if ambiguous, infer from class medians
            x_pos, x_neg = summarize_per_file_for_np(series_list, p, NP_POS_BAND_BELOW, NP_NEG_GAP, NP_NEG_BAND_WIDTH)
            direction = direction_from_slope(b_local) or direction_from_classes(x_pos, x_neg)

            # Thresholds (prefer LOOCV if available)
            theta_pool, tpr_pool, fpr_pool = sweep_np_threshold(x_pos, x_neg, direction, NP_ALPHA_MISS)
            if DO_LOOCV:
                theta_cv, tpr_cv, fpr_cv, n_folds = loocv_np_threshold(series_list, p, direction, NP_ALPHA_MISS,
                                                                       NP_POS_BAND_BELOW, NP_NEG_GAP, NP_NEG_BAND_WIDTH)
            else:
                theta_cv, tpr_cv, fpr_cv, n_folds = (np.nan, np.nan, np.nan, 0)

            # Pick final theta/metrics for output
            if DO_LOOCV and n_folds > 0 and not np.isnan(theta_cv):
                theta_out, TPR_out, FPR_out = theta_cv, tpr_cv, fpr_cv
            else:
                theta_out, TPR_out, FPR_out = theta_pool, tpr_pool, fpr_pool

            rows.append({
                "signal": name,
                "column": col,
                "SoC_point_%": p,
                "detectability": detect,
                "SoC_uncertainty_%": soc_unc,
                "threshold_direction": direction,   # interpret as: trigger low if (x {dir} theta_mV)
                "theta_mV": theta_out,
                "TPR": TPR_out,
                "FPR": FPR_out,
                "N_files": len(series_list),
            })

    if not rows:
        print("No results to write (insufficient files/signals)."); return

    out_df = pd.DataFrame(rows).sort_values(by=["SoC_point_%", "detectability"], ascending=[True, False])
    out_df.to_csv(args.out, index=False)
    print(f"Wrote results to: {args.out}\n")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(out_df.to_string(index=False, float_format=lambda v: f"{v:.6g}"))

if __name__ == "__main__":
    main()
