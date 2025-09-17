#!/usr/bin/env python3
import argparse, re, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TEMP_POINTS = np.array([-20.0, 20.0, 59.0], dtype=float)
THRESHOLDS = {
    "apollo": {"RR": np.array([2739, 2768, 2840], dtype=float),
               "NF": np.array([ 748,  213,   94], dtype=float)},
    "gemini": {"RR": np.array([2766, 2830, 2894], dtype=float),
               "NF": np.array([ 700,  223,   92], dtype=float)},
    "maven":  {"RR": np.array([2753, 2791, 2861], dtype=float),
               "NF": np.array([ 748,  213,   94], dtype=float)},
}

def product_from_serial(sn: str) -> str:
    if not sn: return "gemini"
    c = sn[0].upper()
    if c == "A": return "apollo"
    if c == "B": return "gemini"
    if c == "D": return "maven"
    return "gemini"

def parse_time_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")

def _num(s):
    if pd.isna(s): return np.nan
    if isinstance(s,(int,float,np.floating)): return float(s)
    s = str(s).replace(",", "")
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    return float(m.group(0)) if m else np.nan

def parse_voltage_to_mv(x):
    s = "" if pd.isna(x) else str(x).strip()
    v = _num(s)
    if pd.isna(v): return np.nan
    sl = s.lower()
    if "mv" in sl: return v
    if " v" in sl or sl.endswith("v"): return v * 1000.0
    return v if v > 100 else v * 1000.0

def parse_temp_c(x):
    s = "" if pd.isna(x) else str(x).strip()
    t = _num(s)
    if pd.isna(t): return np.nan
    sl = s.lower()
    if "f" in sl and "c" not in sl: return (t - 32.0) * 5.0 / 9.0
    return t

def interp_thresholds(temp_c: np.ndarray, rr_arr: np.ndarray, nf_arr: np.ndarray):
    rr = np.interp(temp_c, TEMP_POINTS, rr_arr, left=rr_arr[0], right=rr_arr[-1])
    nf = np.interp(temp_c, TEMP_POINTS, nf_arr, left=nf_arr[0], right=nf_arr[-1])
    return rr, nf

def ocv_lowpass_mv(voltage_mv: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    ocv = np.empty_like(voltage_mv, dtype=float)
    prev = np.nan
    for i, v in enumerate(voltage_mv):
        if np.isnan(v):
            ocv[i] = prev
        elif np.isnan(prev):
            ocv[i] = float(int(v + 0.5))
            prev = ocv[i]
        else:
            val = alpha * v + (1.0 - alpha) * prev
            ocv[i] = float(int(val + 0.5))
            prev = ocv[i]
    return ocv

def compute_levels(df: pd.DataFrame, rr_arr: np.ndarray, nf_arr: np.ndarray, alpha: float, boot_comp_mv: int, hyst_count: int) -> pd.DataFrame:
    df = df.copy()
    df["Voltage_mV"] = df["Voltage"].apply(parse_voltage_to_mv).astype(float)
    df["Temperature"] = df["Temperature"].apply(parse_temp_c).astype(float)
    df["OCV_mV"] = ocv_lowpass_mv(df["Voltage_mV"].to_numpy(dtype=float), alpha=alpha)
    rr_thresh, nf_droop = interp_thresholds(df["Temperature"].to_numpy(dtype=float), rr_arr, nf_arr)
    df["RR_Thresh_mV"] = rr_thresh.astype(float)
    df["NF_Droop_Thresh_mV"] = nf_droop.astype(float)
    levels = []
    level = "Unknown"
    rr_c = 0
    initialized = False
    for i in range(len(df)):
        ocv = df.at[i, "OCV_mV"]
        rr_t = df.at[i, "RR_Thresh_mV"]
        if not initialized:
            if np.isnan(ocv) or np.isnan(rr_t):
                levels.append("Unknown")
                continue
            level = "ReplacementRequired" if (ocv + boot_comp_mv) < rr_t else "Good"
            initialized = True
            levels.append(level)
            continue
        if level == "Good":
            if not np.isnan(ocv) and not np.isnan(rr_t) and ocv < rr_t:
                rr_c += 1
                if rr_c >= hyst_count:
                    level = "ReplacementRequired"
                    rr_c = 0
            else:
                rr_c = 0
        levels.append(level)
    df["BatteryLevel"] = levels
    mapping = {"NonFunctional": 0, "ReplacementRequired": 1, "Good": 2, "Unknown": np.nan}
    df["BatteryLevelCode"] = df["BatteryLevel"].map(mapping)
    return df

def read_ops_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    col_time = "time" if "time" in df.columns else "Time" if "Time" in df.columns else None
    col_text = "text" if "text" in df.columns else "Text" if "Text" in df.columns else None
    if not col_time or not col_text:
        raise SystemExit(f"Missing ops columns in {path}")
    df = df.rename(columns={col_time: "time", col_text: "text"})
    df["time"] = parse_time_series(df["time"])
    df["text"] = df["text"].astype(str)
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df

def annotate_ops(ax, ops_aligned, label_fail_only=False):
    if ops_aligned.empty: return
    for _, row in ops_aligned.iterrows():
        t = row["time"]
        y = row["y_at_sample"]
        txt = row["text"]
        is_fail = str(txt).lower() in ["stall","timeout"]
        ax.axvline(t, alpha=0.25, linestyle="--")
        ax.scatter([t], [y], marker="x" if is_fail else "o", zorder=3)
        if is_fail or not label_fail_only:
            ax.text(t, y, txt, rotation=90, va="bottom", ha="center", fontsize=8)

def process_serial(sn: str, data_csv: Path, ops_csv: Path, args):
    df = pd.read_csv(data_csv)
    ren = {}
    if args.time_col != "Time": ren[args.time_col] = "Time"
    if args.temp_col != "Temperature": ren[args.temp_col] = "Temperature"
    if args.volt_col != "Voltage": ren[args.volt_col] = "Voltage"
    df = df.rename(columns=ren)
    if not {"Time","Temperature","Voltage"}.issubset(df.columns):
        missing = {"Time","Temperature","Voltage"} - set(df.columns)
        raise SystemExit(f"{sn}: missing {sorted(missing)} in {data_csv}")
    df["Time"] = parse_time_series(df["Time"])
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    prod = product_from_serial(sn)
    rr_arr = THRESHOLDS[prod]["RR"]
    nf_arr = THRESHOLDS[prod]["NF"]
    out = compute_levels(df, rr_arr, nf_arr, alpha=args.alpha, boot_comp_mv=args.boot_comp_mv, hyst_count=args.hysteresis)
    ops = read_ops_csv(ops_csv) if ops_csv and ops_csv.exists() else pd.DataFrame(columns=["time","text"])
    joined = pd.DataFrame()
    if not ops.empty and not out.empty:
        tmp_out = out[["Time","Voltage_mV","OCV_mV","Temperature","BatteryLevelCode"]].copy()
        tmp_out = tmp_out.rename(columns={"Time":"sample_time"})
        tol = pd.Timedelta(seconds=args.tolerance_sec)
        ops_sorted = ops.sort_values("time")
        tmp_out_sorted = tmp_out.sort_values("sample_time")
        m = pd.merge_asof(ops_sorted, tmp_out_sorted, left_on="time", right_on="sample_time", direction="nearest", tolerance=tol)
        m = m.dropna(subset=["sample_time"]).reset_index(drop=True)
        m["y_voltage"] = m["OCV_mV"]
        m["y_temp"] = m["Temperature"]
        m["y_level"] = m["BatteryLevelCode"]
        joined = m
    fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axs[0].plot(out["Time"], out["Voltage_mV"], label="Voltage (mV)")
    axs[0].plot(out["Time"], out["OCV_mV"], label="OCV (mV)")
    axs[0].set_ylabel("mV")
    axs[0].set_title(f"{sn} — {prod.capitalize()} — Voltage")
    axs[0].legend()
    if not joined.empty:
        ja = joined.copy()
        ja["y_at_sample"] = ja["y_voltage"]
        annotate_ops(axs[0], ja[["time","text","y_at_sample"]], label_fail_only=True)
    axs[1].plot(out["Time"], out["Temperature"])
    axs[1].set_ylabel("°C")
    axs[1].set_title("Temperature")
    if not joined.empty:
        jb = joined.copy()
        jb["y_at_sample"] = jb["y_temp"]
        annotate_ops(axs[1], jb[["time","text","y_at_sample"]], label_fail_only=True)
    axs[2].step(out["Time"], out["BatteryLevelCode"], where="post")
    axs[2].set_yticks([0,1,2]); axs[2].set_yticklabels(["NF","RR","Good"]); axs[2].set_ylim(-0.5,2.5)
    axs[2].set_xlabel("Time"); axs[2].set_ylabel("Level"); axs[2].set_title("Battery Level")
    if not joined.empty:
        jc = joined.copy()
        jc["y_at_sample"] = jc["y_level"].clip(-0.49, 2.49)
        annotate_ops(axs[2], jc[["time","text","y_at_sample"]], label_fail_only=False)
    plt.tight_layout()
    outdir = Path(args.out_dir) if args.out_dir else data_csv.parent
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / f"{sn}_battery_plots.png"
    plt.savefig(png_path, dpi=150)
    if args.show: plt.show()
    plt.close(fig)
    if args.save_csv:
        out.to_csv(outdir / f"{sn}_augmented.csv", index=False)
        if not joined.empty:
            joined.to_csv(outdir / f"{sn}_ops_aligned.csv", index=False)
    return str(png_path)

def find_pairs(folder: Path):
    files = list(folder.glob("*.csv"))
    data = {}
    ops = {}
    for p in files:
        name = p.name
        if name.endswith("_ops.csv"):
            base = name[:-8]
            ops[base] = p
        else:
            base = p.stem
            data[base] = p
    sns = sorted(set(data.keys()) & set(ops.keys()))
    return [(sn, data[sn], ops[sn]) for sn in sns]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--serials", nargs="*", default=None)
    ap.add_argument("--time-col", default="Time")
    ap.add_argument("--temp-col", default="Temperature")
    ap.add_argument("--volt-col", default="Voltage")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--boot-comp-mv", type=int, default=25)
    ap.add_argument("--hysteresis", type=int, default=3)
    ap.add_argument("--tolerance-sec", type=int, default=1800)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--save-csv", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists(): raise SystemExit(f"Folder not found: {folder}")
    pairs = find_pairs(folder)
    if args.serials:
        pairs = [t for t in pairs if t[0] in set(args.serials)]
    if not pairs: raise SystemExit("No SN pairs (SN.csv + SN_ops.csv) found.")
    outputs = []
    for sn, data_csv, ops_csv in pairs:
        outputs.append(process_serial(sn, data_csv, ops_csv, args))
    for p in outputs:
        print(p)

if __name__ == "__main__":
    main()
