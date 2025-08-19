#!/usr/bin/env python3
import os
import re
import glob
import math
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict

CSV_DIR   = "csvs/cross_temp"
TIME_COL  = "Time Elapsed (hours)"
BATT_COL  = "batt_mV"

FAULT_COLS = [
    "fault_brownout",
    "fault_sound",
    "fault_sound_brownout",
    "fault_bolt",
    "fault_bolt_brownout",
]

RANDOM_STATE = 42
MAX_ITER     = 5000
HIDDEN       = (32, 32)
VAL_FRAC     = 0.15
MIN_SAMPLES_PER_TEMP = 40
EPS = 1e-4

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
import matplotlib.pyplot as plt

def parse_temperature_from_folder(name: str) -> Optional[float]:
    parts = name.split('-')
    if len(parts) < 2:
        return None
    temp_str = parts[1].lower()
    if temp_str.startswith("minus"):
        try:
            return -float(temp_str.replace("minus", ""))
        except Exception:
            return None
    try:
        return float(temp_str)
    except Exception:
        return None

def first_fault_label(df: pd.DataFrame) -> Optional[pd.Index]:
    present_fault_cols = [c for c in FAULT_COLS if c in df.columns]
    if not present_fault_cols:
        return None
    any_fault = df[present_fault_cols].any(axis=1)
    if not any_fault.any():
        return None
    n_rows = len(df)
    min_idx = int(np.ceil(0.10 * n_rows))
    if min_idx > 0:
        any_fault.iloc[:min_idx] = False
    if not any_fault.any():
        return None
    return any_fault.idxmax()

def collect_voltage_soc_pairs(csv_path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if TIME_COL not in df.columns or BATT_COL not in df.columns:
        return None
    idx_fault = first_fault_label(df)
    if idx_fault is None:
        return None
    t_series = df[TIME_COL]
    t0 = float(t_series.iloc[0])
    t_fault = float(t_series.loc[idx_fault])
    if not np.isfinite(t0) or not np.isfinite(t_fault) or t_fault <= t0:
        return None
    try:
        pos_fault = df.index.get_indexer([idx_fault])[0]
    except Exception:
        return None
    df_use = df.iloc[:pos_fault + 1].copy()
    t = df_use[TIME_COL].astype(float)
    soc = (t_fault - t) / (t_fault - t0)
    v = df_use[BATT_COL].astype(float)
    mask = np.isfinite(v) & np.isfinite(soc)
    v = v[mask].values.reshape(-1, 1)
    s = soc[mask].values
    if len(s) < 5:
        return None
    return pd.DataFrame({"voltage_mV": v.flatten(), "soc": s})

def aggregate_by_temperature(root_dir: str) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    subfolders = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )
    for sf in subfolders:
        folder_path = os.path.join(root_dir, sf)
        csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        if not csv_files:
            continue
        rows = []
        temps = []
        for p in csv_files:
            pairs = collect_voltage_soc_pairs(p)
            if pairs is not None:
                rows.append(pairs)
                try:
                    df = pd.read_csv(p)
                    for col in ["temperature", "temp", "Temperature", "Temp"]:
                        if col in df.columns:
                            temp_vals = pd.to_numeric(df[col], errors="coerce").dropna()
                            if not temp_vals.empty:
                                temps.extend(temp_vals.tolist())
                                break
                except Exception:
                    pass
        if not rows:
            continue
        df_all = pd.concat(rows, ignore_index=True)
        df_all = df_all.dropna()
        df_all = df_all.sort_values("voltage_mV")
        df_all = df_all.reset_index(drop=True)
        if len(df_all) < MIN_SAMPLES_PER_TEMP:
            continue
        vmin = float(np.nanmin(df_all["voltage_mV"].values))
        vmax = float(np.nanmax(df_all["voltage_mV"].values))
        temp_value = None
        if temps:
            temps_arr = np.array(temps)
            vals, counts = np.unique(temps_arr, return_counts=True)
            if len(vals) > 0:
                mode_idx = np.argmax(counts)
                temp_value = float(vals[mode_idx])
        if temp_value is None:
            temp_value = parse_temperature_from_folder(sf)
        out[sf] = {
            "temp_value": temp_value,
            "data": df_all,
            "vmin": vmin,
            "vmax": vmax,
        }
    return out

def _logit_safe(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, EPS, 1 - EPS)
    return np.log(y / (1 - y))

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def fit_voltage_to_soc_nn(df: pd.DataFrame):
    X = df[["voltage_mV"]].values
    y = df["soc"].values
    Xtr, Xval, ytr, yval = train_test_split(
        X, y, test_size=VAL_FRAC, random_state=RANDOM_STATE, shuffle=True
    )
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=HIDDEN,
            activation="relu",
            random_state=RANDOM_STATE,
            max_iter=MAX_ITER,
            n_iter_no_change=50,
            learning_rate_init=1e-3,
            early_stopping=True,
            validation_fraction=0.15,
            tol=1e-5,
        )),
    ])
    model = TransformedTargetRegressor(
        regressor=base,
        func=_logit_safe,
        inverse_func=_sigmoid,
        check_inverse=False,
    )
    model.fit(Xtr, ytr)
    return model

def find_voltage_for_soc(model, soc_target: float, v_lo: float, v_hi: float,
                         max_iter: int = 60, tol: float = 1e-3) -> Optional[float]:
    def f(v):
        pred = float(model.predict(np.array(v, ndmin=2).reshape(-1, 1))[0])
        pred = float(np.clip(pred, 0.0, 1.0))
        return pred - soc_target
    flo = f(v_lo)
    fhi = f(v_hi)
    span = (v_hi - v_lo)
    if flo * fhi > 0:
        for _ in range(5):
            v_lo = v_lo - 0.02 * span
            v_hi = v_hi + 0.02 * span
            flo = f(v_lo)
            fhi = f(v_hi)
            if flo * fhi <= 0:
                break
    if flo * fhi > 0:
        return None
    lo, hi = v_lo, v_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol:
            return mid
        if flo * fmid <= 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi)

def plot_nn_outputs(buckets):
    plt.figure(figsize=(10, 6))
    for name, payload in buckets.items():
        df = payload["data"]
        vmin = payload["vmin"]
        vmax = payload["vmax"]
        model = fit_voltage_to_soc_nn(df)
        voltages = np.linspace(vmin, vmax, 200).reshape(-1, 1)
        soc_pred = model.predict(voltages)
        soc_pred = np.clip(soc_pred, 0.0, 1.0)
        label = f"{payload['temp_value']}" if payload["temp_value"] is not None else name
        plt.plot(voltages.flatten(), soc_pred, label=label)
    plt.xlabel("Voltage (mV)")
    plt.ylabel("Predicted SoC")
    plt.title("NN Output: SoC vs. Voltage by Temperature")
    plt.legend(title="Temperature")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    if not os.path.isdir(CSV_DIR):
        print(f"Missing directory: {CSV_DIR}")
        return
    buckets = aggregate_by_temperature(CSV_DIR)
    if not buckets:
        print(f"No usable data found under {CSV_DIR}")
        return
    def sort_key(item):
        name, payload = item
        tv = payload["temp_value"]
        return (0, tv) if tv is not None else (1, name)
    print(f"Found {len(buckets)} temperature buckets with sufficient data.\n")
    print("Voltage at target SoC by temperature")
    print("===================================")
    print(f"{'Temp':>10} | {'V@10% SoC (mV)':>16} | {'V@1% SoC (mV)':>15} | {'Samples':>7}")
    print("-" * 58)
    for name, payload in sorted(buckets.items(), key=sort_key):
        df = payload["data"]
        vmin = payload["vmin"]
        vmax = payload["vmax"]
        model = fit_voltage_to_soc_nn(df)
        v_at_10 = find_voltage_for_soc(model, soc_target=0.10, v_lo=vmin, v_hi=vmax)
        v_at_01 = find_voltage_for_soc(model, soc_target=0.01, v_lo=vmin, v_hi=vmax)
        label = f"{payload['temp_value']}" if payload["temp_value"] is not None else name
        v10_str = f"{v_at_10:.2f}" if v_at_10 is not None else "—"
        v01_str = f"{v_at_01:.2f}" if v_at_01 is not None else "—"
        print(f"{label:>10} | {v10_str:>16} | {v01_str:>15} | {len(df):>7}")
    print("\nNotes:")
    print("- SoC is inferred from time-to-first-fault within each CSV (normalized 1→0).")
    print("- We train one NN per temperature bucket (subfolder) to learn SoC ≈ f(voltage).")
    print("- Voltages for 10% and 1% SoC are found by inverting the NN via bisection.")
    plot_nn_outputs(buckets)

if __name__ == "__main__":
    main()
