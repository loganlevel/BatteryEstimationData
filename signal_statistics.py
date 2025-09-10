#!/usr/bin/env python3

import argparse, os, glob
import pandas as pd
import numpy as np
from collections import defaultdict

# =========================
# =======  CONFIG  ========
# =========================

# Estimated lifetime (days) by product and days-left targets you care about
ESTIMATED_LIFE_DAYS = {
    "Lock Pro": 207,
    "Lock Plus": 139,
    "Bolt": 323,
}
DAYS_LEFT_TARGETS = {
    "Lock Pro": [25, 4],
    "Lock Plus": [25, 4],
    "Bolt": [25, 4],
}

# Compute SoC targets from the days-left targets (ceil to be conservative)
TARGET_SOC_POINTS = sorted(set(
    float(np.ceil(100.0 * (days_left / ESTIMATED_LIFE_DAYS[prod])))
    for prod, targets in DAYS_LEFT_TARGETS.items()
    for days_left in targets
))

# Local slope & sigma windows (in %SoC)
SLOPE_WINDOW_PCT = 10.0
SIGMA_BAND_PCT   = 2.0

# 0% handling convention for slope/σ windows
# "edge" -> center at 0% (windows hug edge); "mid" -> center at half-width (stable)
ZERO_SLOPE_CENTER = "mid"   # "edge" or "mid"
ZERO_SIGMA_CENTER = "mid"   # "edge" or "mid"

# Fault search / noise / dataset requirements
IGNORE_STARTUP_FRAC = 0.10  # ignore first 10% rows when finding first fault
ADC_NOISE_MV        = 0.0   # meas noise added in quadrature to cross-file spread
MIN_FILES           = 3     # min CSVs per signal to report

# Neyman–Pearson thresholding (safety-first)
NP_ALPHA_MISS       = 0.01  # require TPR >= 1 - alpha
NP_POS_BAND_BELOW   = 2.0   # pos class: [max(0, p - this), p]
NP_NEG_GAP          = 5.0   # exclude [p, p+gap]
NP_NEG_BAND_WIDTH   = 10.0  # neg class: [p+gap, p+gap+width]

# Cross-validation
DO_LOOCV            = True  # prefer LOOCV median theta; fallback to pooled

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

def get_time_hours(df, fault_idx):
    # Prefer explicit time if present; else return None (time metrics become NaN)
    col = None
    for c in df.columns:
        if c.strip().lower() in ("time elapsed (hours)", "time_elapsed_hours", "elapsed_hours"):
            col = c
            break
    if col is None:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").values[:fault_idx+1]
    return vals.astype(float)

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
    t_hours = get_time_hours(df, fault_idx)  # may be None
    return soc, x, t_hours

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
    for soc, x, _ in per_trial_series:
        m = (soc >= lo) & (soc <= hi)
        if m.sum() == 0: continue
        vals.append(float(np.nanmean(x[m])))
    if len(vals) < 2: return np.nan
    between = np.nanstd(vals, ddof=1)
    return float(np.sqrt(between**2 + adc_noise**2))

def build_pooled_arrays(per_trial_series):
    if not per_trial_series: return np.array([]), np.array([])
    soc_all = np.concatenate([s for s,_,_ in per_trial_series])
    x_all   = np.concatenate([x for _,x,_ in per_trial_series])
    return soc_all, x_all

def class_bands(soc, p, pos_below, neg_gap, neg_width):
    pos_lo = max(0.0, p - pos_below); pos_hi = p
    neg_lo = min(100.0, p + neg_gap); neg_hi = min(100.0, p + neg_width + neg_gap)
    m_pos = (soc >= pos_lo) & (soc <= pos_hi)
    m_neg = (soc >= neg_lo) & (soc <= neg_hi)
    return m_pos, m_neg

def summarize_per_file_for_np(series_list, p, pos_below, neg_gap, neg_width):
    x_pos, x_neg = [], []
    for soc, x, _ in series_list:
        m_pos, m_neg = class_bands(soc, p, pos_below, neg_gap, neg_width)
        if m_pos.any(): x_pos.append(float(np.nanmean(x[m_pos])))
        if m_neg.any(): x_neg.append(float(np.nanmean(x[m_neg])))
    return np.array(x_pos), np.array(x_neg)

def direction_from_slope(b_local):
    if np.isnan(b_local) or abs(b_local) < 1e-12:
        return None
    return (">=") if (b_local < 0) else ("<=")

def direction_from_classes(x_pos, x_neg):
    if x_pos.size == 0 or x_neg.size == 0: return "<="
    mpos = np.median(x_pos); mneg = np.median(x_neg)
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

def first_trigger_index(x, direction, theta):
    if direction == ">=":
        idxs = np.where(x >= theta)[0]
    else:
        idxs = np.where(x <= theta)[0]
    return int(idxs[0]) if idxs.size > 0 else None

def time_metrics_for_threshold(series_list, direction, theta, p):
    """
    For each file, compute how many days early/late the trigger is relative to the target lead-time
    implied by SoC point p. The target lead-time for a given file is (p/100)*total_life_days.
    Returns (early_mean, early_worst, late_mean, late_worst, frac_late, n_used)
    """
    early_days = []
    late_days  = []
    used = 0
    for soc, x, t_hours in series_list:
        if t_hours is None or len(t_hours) < 2:
            # Can't compute time metrics without elapsed hours
            continue
        used += 1
        t_dead_h = float(t_hours[-1])
        total_days = t_dead_h / 24.0
        target_days = (p / 100.0) * total_days  # desired days before dead for this file

        trig_idx = first_trigger_index(x, direction, theta)
        if trig_idx is None:
            # Never triggered before dead -> fully late by target_days
            late_days.append(target_days)
            continue

        t_trig_h = float(t_hours[trig_idx])
        days_left_at_trigger = max(0.0, (t_dead_h - t_trig_h) / 24.0)
        delta = days_left_at_trigger - target_days  # + = early, - = late

        if delta >= 0:
            early_days.append(delta)
        else:
            late_days.append(-delta)

    def mean_or_zero(a):
        return float(np.mean(a)) if len(a) else 0.0
    def max_or_zero(a):
        return float(np.max(a)) if len(a) else 0.0

    early_mean = mean_or_zero(early_days)
    early_worst = max_or_zero(early_days)
    late_mean = mean_or_zero(late_days)
    late_worst = max_or_zero(late_days)
    n_total = used if used > 0 else 1
    frac_late = float(len(late_days)) / float(n_total)
    return early_mean, early_worst, late_mean, late_worst, frac_late, used

def main():
    parser = argparse.ArgumentParser(description="Compute minimal detectability & thresholds + time metrics.")
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
                soc, x, t_hours = per_file_series(df, col, fi)
                if len(soc) >= 3: series_by_signal[(name, col)].append((soc, x, t_hours))
            except Exception as e:
                print(f"[WARN] Failed parsing series in {pth} ({col}): {e}")

    rows = []
    for (name, col), series_list in series_by_signal.items():
        if len(series_list) < MIN_FILES: continue
        soc_all, x_all = build_pooled_arrays(series_list)

        for p in TARGET_SOC_POINTS:
            # Centers per convention
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

            # Direction: slope sign first; fall back to class medians
            x_pos, x_neg = summarize_per_file_for_np(series_list, p, NP_POS_BAND_BELOW, NP_NEG_GAP, NP_NEG_BAND_WIDTH)
            direction = direction_from_slope(b_local) or direction_from_classes(x_pos, x_neg)

            # Thresholds (prefer LOOCV)
            theta_pool, tpr_pool, fpr_pool = sweep_np_threshold(x_pos, x_neg, direction, NP_ALPHA_MISS)
            if DO_LOOCV:
                theta_cv, tpr_cv, fpr_cv, n_folds = loocv_np_threshold(series_list, p, direction, NP_ALPHA_MISS,
                                                                       NP_POS_BAND_BELOW, NP_NEG_GAP, NP_NEG_BAND_WIDTH)
            else:
                theta_cv, tpr_cv, fpr_cv, n_folds = (np.nan, np.nan, np.nan, 0)

            if DO_LOOCV and n_folds > 0 and not np.isnan(theta_cv):
                theta_out, TPR_out, FPR_out = theta_cv, tpr_cv, fpr_cv
            else:
                theta_out, TPR_out, FPR_out = theta_pool, tpr_pool, fpr_pool

            # --- Time-domain evaluation of the chosen threshold (days early/late) ---
            early_mean, early_worst, late_mean, late_worst, frac_late, n_used = \
                time_metrics_for_threshold(series_list, direction, theta_out, p)

            rows.append({
                "signal": name,
                "column": col,
                "SoC_point_%": int(np.floor(p)),
                "detectability": round(detect, 3) if not np.isnan(detect) else np.nan,
                "SoC_uncertainty_%": round(soc_unc, 2) if not np.isnan(soc_unc) else np.nan,
                "threshold_direction": direction,     # interpret: trigger if (x {dir} theta_mV)
                "theta_mV": int(np.floor(theta_out)) if not np.isnan(theta_out) else np.nan,
                "TPR": round(TPR_out, 3) if not np.isnan(TPR_out) else np.nan,
                "FPR": round(FPR_out, 3) if not np.isnan(FPR_out) else np.nan,
                "N_files": int(len(series_list)),
                # Time-domain metrics (days); computed when Time Elapsed (hours) is present
                "avg_days_early": round(early_mean, 2) if not np.isnan(early_mean) else np.nan,
                "worst_days_early": round(early_worst, 2) if not np.isnan(early_worst) else np.nan,
                "avg_days_late": round(late_mean, 2) if not np.isnan(late_mean) else np.nan,
                "worst_days_late": round(late_worst, 2) if not np.isnan(late_worst) else np.nan,
                "frac_late": round(frac_late, 3) if not np.isnan(frac_late) else np.nan,
                "N_files_with_time": int(n_used),
            })

    if not rows:
        print("No results to write (insufficient files/signals)."); return

    out_df = pd.DataFrame(rows).sort_values(by=["SoC_point_%", "detectability"], ascending=[True, False])
    out_df.to_csv(args.out, index=False)
    print(f"Wrote results to: {args.out}\n")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(out_df.to_string(index=False, float_format=lambda v: f"{v:.6g}"))

if __name__ == "__main__":
    main()
