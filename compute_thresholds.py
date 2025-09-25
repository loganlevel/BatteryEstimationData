#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emit C arrays of temperature-dependent thresholds (copy/paste ready).

- First-level subfolders under the root are treated as "temperature buckets".
- Bucket temperature is the mode of per-file median 'temp' (°C), rounded to int.
- For each product and bucket:
    * Upper SoC target (days-left -> SoC): OCV threshold (time-constrained if time available; else NP fallback).
    * Lower SoC target: ST droop threshold (NP backstop).
- Prints:
    #if defined(CONFIG_PRODUCT_XXX)
    static const BatteryThresholdVoltages_t temperatureDependentBatteryThresholds[] = {
        { .thresholdTemp = T, .replacementRequiredOCVThreshold = <mV>, .nonFunctionalSTDroopThreshold = <mV> },
        ...
    };
    #endif
"""

import os, glob
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# =========================
# =======  CONFIG  ========
# =========================

ESTIMATED_LIFE_DAYS = {"Maven": 207, "Gemini": 139, "Apollo": 323}
DAYS_LEFT_TARGETS   = {"Maven": [25, 1], "Gemini": [25, 1], "Apollo": [25, 1]}  # [upper, lower] days-left

# Mapping Product -> C macro name
PRODUCT_MACROS = {"Maven": "CONFIG_PRODUCT_MAVEN",
                  "Gemini": "CONFIG_PRODUCT_GEMINI",
                  "Apollo": "CONFIG_PRODUCT_APOLLO"}

# Fault search / SoC mapping
IGNORE_STARTUP_FRAC = 0.30  # skip first 30% rows when searching for first fault
MIN_ROWS_FOR_SERIES = 3

# Time-constrained selection (for OCV @ upper target)
TIME_ALPHA_LATE = 0.01      # ≤1% late
TIME_OBJECTIVE  = "median"  # minimize “median-like” earliness
USE_TIME_CONSTRAINED_IF_TIME_AVAILABLE = True

# Neyman–Pearson selection (for ST @ lower target)
NP_ALPHA_MISS     = 0.005
NP_POS_BAND_BELOW = 1.0
NP_NEG_GAP        = 5.0
NP_NEG_BAND_WIDTH = 10.0

# --- ST conservatism controls (apply only to ST @ lower target) ---
ST_USE_NEG_PERCENTILE = True      # force theta above a high percentile of healthy
ST_NEG_PERCENTILE     = 99.9      # e.g., 95th percentile of x_neg for direction ">="
ST_SAFETY_MARGIN_MV   = 0.0      # fixed upward bump in mV (tune 10–40 mV)


# Rounding
ROUND_THETA_MV = 0  # integer mV

# =========================
# =======  HELPERS  =======
# =========================

def list_buckets(root):
    return [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]

def find_csvs(root):
    return [p for p in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)]

def load_csv(path):
    df = pd.read_csv(path)
    # Signals
    for c in ("batt_mV", "soundDroopMag_mV", "soundDroop_mV", "temp"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Faults
    if "faults" in df.columns:
        df["faults"] = pd.to_numeric(df["faults"], errors="coerce").fillna(0).astype(int)
    for c in df.columns:
        if str(c).startswith("fault_"):
            df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    # Time
    for k in ("Time Elapsed (hours)", "time_elapsed_hours", "elapsed_hours"):
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
            break
    return df

def first_fault_index(df, ignore_frac):
    n = len(df)
    if n == 0: return None
    start = int(np.floor(ignore_frac * n))
    cond = np.zeros(n, dtype=bool)
    if "faults" in df.columns:
        cond |= (df["faults"].to_numpy() != 0)
    for c in df.columns:
        if str(c).startswith("fault_"):
            cond |= df[c].astype(bool).to_numpy()
    rel = np.where(cond[start:])[0]
    if rel.size: return start + rel[0]
    return n - 1  # treat EOF as "dead" if no explicit fault

def soc_vector(length, fault_idx):
    idx = np.arange(fault_idx + 1, dtype=float)
    return 100.0 * (1.0 - idx / float(fault_idx))

def get_time_hours(df, fault_idx):
    for k in ("Time Elapsed (hours)", "time_elapsed_hours", "elapsed_hours"):
        if k in df.columns:
            return df[k].to_numpy(dtype=float)[:fault_idx+1]
    return None

def pick_signal_columns(df):
    cols = {}
    cols["OCV"] = "batt_mV" if "batt_mV" in df.columns else None
    if "soundDroopMag_mV" in df.columns:
        cols["ST"] = "soundDroopMag_mV"
    elif "soundDroop_mV" in df.columns:
        cols["ST"] = "soundDroop_mV"
    else:
        cols["ST"] = None
    return {k: v for k, v in cols.items() if v is not None}

def per_file_series(df, sig_col, fi):
    x = df[sig_col].to_numpy(dtype=float)[:fi+1]
    soc = soc_vector(len(x), fi)
    t_hours = get_time_hours(df, fi)
    return soc, x, t_hours

def class_bands(soc, p, pos_below, neg_gap, neg_width):
    pos_lo = max(0.0, p - pos_below); pos_hi = p
    neg_lo = min(100.0, p + neg_gap); neg_hi = min(100.0, p + neg_gap + neg_width)
    return (soc >= pos_lo) & (soc <= pos_hi), (soc >= neg_lo) & (soc <= neg_hi)

def summarize_per_file_for_np(series_list, p):
    x_pos, x_neg = [], []
    for soc, x, _ in series_list:
        m_pos, m_neg = class_bands(soc, p, NP_POS_BAND_BELOW, NP_NEG_GAP, NP_NEG_BAND_WIDTH)
        if m_pos.any(): x_pos.append(float(np.nanmean(x[m_pos])))
        if m_neg.any(): x_neg.append(float(np.nanmean(x[m_neg])))
    return np.array(x_pos), np.array(x_neg)

def direction_from_classes(series_list, p):
    x_pos, x_neg = summarize_per_file_for_np(series_list, p)
    if x_pos.size == 0 or x_neg.size == 0:
        return None
    return ">=" if np.median(x_pos) > np.median(x_neg) else "<="

def sweep_np_threshold(x_pos, x_neg, direction, alpha):
    if x_pos.size == 0 or x_neg.size == 0 or direction is None:
        return (np.nan, np.nan, np.nan)
    xs = np.unique(np.concatenate([x_pos, x_neg]))
    if xs.size == 0: return (np.nan, np.nan, np.nan)
    span = (xs[-1] - xs[0]) if xs.size > 1 else 1.0
    eps = 1e-9 * span
    cands = []
    for th in xs:
        if direction == ">=":
            tpr = np.mean(x_pos >= th); fpr = np.mean(x_neg >= th)
            tpr2 = np.mean(x_pos >= (th - eps)); fpr2 = np.mean(x_neg >= (th - eps))
        else:
            tpr = np.mean(x_pos <= th); fpr = np.mean(x_neg <= th)
            tpr2 = np.mean(x_pos <= (th + eps)); fpr2 = np.mean(x_neg <= (th + eps))
        cands.append((th, tpr, fpr)); cands.append(((th - eps) if direction==">=" else (th + eps), tpr2, fpr2))
    feasible = [(th,tpr,fpr) for th,tpr,fpr in cands if tpr >= (1.0 - alpha)]
    if feasible:
        feasible.sort(key=lambda z: (z[2], -z[1], -z[0])); return feasible[0]
    max_tpr = max(z[1] for z in cands)
    near = [z for z in cands if abs(z[1] - max_tpr) < 1e-12]
    near.sort(key=lambda z: z[2]); return near[0]

def first_trigger_index(x, direction, theta):
    if np.isnan(theta) or direction is None: return None
    idxs = np.where(x >= theta)[0] if direction == ">=" else np.where(x <= theta)[0]
    return int(idxs[0]) if idxs.size > 0 else None

def time_metrics_for_threshold(series_list, direction, theta, p):
    early_days, late_days, used = [], [], 0
    for soc, x, t_hours in series_list:
        if t_hours is None or len(t_hours) < 2:
            continue
        used += 1
        t_dead_h = float(t_hours[-1]); total_days = t_dead_h / 24.0
        target_days = (p / 100.0) * total_days
        ti = first_trigger_index(x, direction, theta)
        if ti is None:
            late_days.append(target_days); continue
        t_trig_h = float(t_hours[ti])
        days_left_at_trigger = max(0.0, (t_dead_h - t_trig_h)/24.0)
        delta = days_left_at_trigger - target_days
        (early_days if delta >= 0 else late_days).append(abs(delta))
    def mean0(a): return float(np.mean(a)) if a else 0.0
    n_total = used if used > 0 else 1
    frac_late = float(len(late_days)) / float(n_total)
    return mean0(early_days), (max(early_days) if early_days else 0.0), \
           mean0(late_days), (max(late_days) if late_days else 0.0), \
           frac_late, used

def select_theta_time_constrained(series_list, p, alpha_late=0.01, objective="median"):
    direction = direction_from_classes(series_list, p)
    x_pos, x_neg = summarize_per_file_for_np(series_list, p)
    cands = np.unique(np.concatenate([x_pos, x_neg])) if (x_pos.size or x_neg.size) else np.array([])
    if cands.size == 0:
        xs_all = np.unique(np.concatenate([x for _,x,_ in series_list])) if series_list else np.array([])
        if xs_all.size == 0: return (np.nan, direction, "no_candidates")
        k = max(1, xs_all.size // 200); cands = xs_all[::k]
    best = None; best_note = None
    for th in cands:
        e_mean, e_worst, l_mean, l_worst, frac_late, n_used = time_metrics_for_threshold(series_list, direction, th, p)
        if n_used == 0:
            best_note = "no_time_in_bucket"; continue
        if frac_late > alpha_late:
            continue
        score = np.median([e_mean, e_worst]) if objective=="median" else (e_worst if objective=="worst" else e_mean)
        cand = (score, e_worst, th)
        if (best is None) or (cand < best): best = cand
    if best is None:
        return (np.nan, direction, best_note or "no_feasible_time_theta")
    _, _, th = best
    return (float(th), direction, "time_constrained")

def select_theta_np(series_list, p):
    direction = direction_from_classes(series_list, p)
    x_pos, x_neg = summarize_per_file_for_np(series_list, p)
    th, tpr, fpr = sweep_np_threshold(x_pos, x_neg, direction, NP_ALPHA_MISS)

    # --- FORCE CONSERVATISM FOR ST-STYLE SIGNALS ---
    # We don't know the caller’s signal name here, so apply a generic conservative push.
    # For droop signals (direction == ">="), "more conservative" means a *higher* theta.
    # For voltage-like (direction == "<="), "more conservative" means a *lower* theta.
    if np.isfinite(th) and direction is not None:
        if ST_USE_NEG_PERCENTILE and x_neg.size > 0:
            if direction == ">=":
                # push above high end of healthy distribution
                th = max(th, float(np.percentile(x_neg, ST_NEG_PERCENTILE)))
            else:
                # push below low end for "<=" signals
                th = min(th, float(np.percentile(x_neg, 100.0 - ST_NEG_PERCENTILE)))

        # add fixed safety margin
        if direction == ">=":
            th = th + ST_SAFETY_MARGIN_MV
        else:
            th = th - ST_SAFETY_MARGIN_MV

    return (float(th) if np.isfinite(th) else np.nan, direction, "np_backstop")


def per_file_median_temp(df):
    if "temp" not in df.columns: return None
    v = df["temp"].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0: return None
    return float(np.median(v))

def bucket_threshold_temp_c(bucket_path):
    temps = []
    for p in find_csvs(bucket_path):
        try:
            df = load_csv(p)
        except Exception:
            continue
        tm = per_file_median_temp(df)
        if tm is not None:
            temps.append(int(np.rint(tm)))
    if not temps:
        return None
    # mode of rounded per-file medians
    cnt = Counter(temps).most_common()
    mode_temp = cnt[0][0]
    return mode_temp

# =========================
# ========== MAIN =========
# =========================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Emit C arrays of OCV/ST thresholds per temperature bucket.")
    parser.add_argument("root", help="Root folder with first-level bucket subfolders.")
    args = parser.parse_args()

    buckets = sorted(list_buckets(args.root))
    # Gather per-bucket series
    bucket_series = {}  # bucket_name -> { "temp_c": int or None, "signals": { "OCV": [...], "ST": [...] } }
    for b in buckets:
        bname = os.path.basename(b)
        temp_c = bucket_threshold_temp_c(b)
        signals = defaultdict(list)
        for p in find_csvs(b):
            try:
                df = load_csv(p)
            except Exception:
                continue
            fi = first_fault_index(df, IGNORE_STARTUP_FRAC)
            if fi is None or fi < 2: continue
            for sig, col in pick_signal_columns(df).items():
                try:
                    soc, x, t_hours = per_file_series(df, col, fi)
                    if len(soc) >= MIN_ROWS_FOR_SERIES:
                        signals[sig].append((soc, x, t_hours))
                except Exception:
                    continue
        bucket_series[bname] = {"temp_c": temp_c, "signals": signals}

    # Compute thresholds per product x bucket
    per_product_entries = {prod: [] for prod in ESTIMATED_LIFE_DAYS.keys()}
    for bname in sorted(bucket_series.keys(), key=lambda k: (bucket_series[k]["temp_c"] is None, bucket_series[k]["temp_c"], k)):
        temp_c = bucket_series[bname]["temp_c"]
        signals = bucket_series[bname]["signals"]
        ocv_series = signals.get("OCV", [])
        st_series  = signals.get("ST", [])

        for prod in ESTIMATED_LIFE_DAYS.keys():
            life_days = ESTIMATED_LIFE_DAYS[prod]
            upper_days, lower_days = DAYS_LEFT_TARGETS[prod]
            upper_soc = float(np.ceil(100.0 * (upper_days / life_days)))
            lower_soc = float(np.ceil(100.0 * (lower_days / life_days)))

            # OCV @ upper
            ocv_theta = np.nan; ocv_dir = None
            if ocv_series:
                if USE_TIME_CONSTRAINED_IF_TIME_AVAILABLE:
                    ocv_theta, ocv_dir, note = select_theta_time_constrained(ocv_series, upper_soc, TIME_ALPHA_LATE, TIME_OBJECTIVE)
                    if not np.isfinite(ocv_theta):
                        ocv_theta, ocv_dir, note = select_theta_np(ocv_series, upper_soc)
                else:
                    ocv_theta, ocv_dir, note = select_theta_np(ocv_series, upper_soc)

            # ST @ lower
            st_theta = np.nan; st_dir = None
            if st_series:
                st_theta, st_dir, _ = select_theta_np(st_series, lower_soc)

            # Only emit entries with numeric thresholds and known temp
            if np.isfinite(ocv_theta) and np.isfinite(st_theta) and (temp_c is not None):
                per_product_entries[prod].append({
                    "temp": int(temp_c),
                    "ocv_mV": int(np.rint(ocv_theta)) if ROUND_THETA_MV == 0 else round(float(ocv_theta), ROUND_THETA_MV),
                    "st_mV":  int(np.rint(st_theta))  if ROUND_THETA_MV == 0 else round(float(st_theta),  ROUND_THETA_MV),
                })

    # Print C blocks
    for idx, prod in enumerate(("Maven", "Gemini", "Apollo")):
        macro = PRODUCT_MACROS[prod]
        entries = sorted(per_product_entries[prod], key=lambda r: r["temp"])
        if idx == 0:
            print(f"#if defined({macro})")
        else:
            print(f"#elif defined({macro})")
        print("static const BatteryThresholdVoltages_t temperatureDependentBatteryThresholds[] =")
        print("{")
        for i, r in enumerate(entries):
            print("    {")
            print(f"        .thresholdTemp = {r['temp']},")
            print(f"        .replacementRequiredOCVThreshold = {r['ocv_mV']},")
            print(f"        .nonFunctionalSTDroopThreshold = {r['st_mV']}")
            print("    }" + ("," if i < len(entries)-1 else ""))
        print("};")
    print("#endif")

if __name__ == "__main__":
    main()
