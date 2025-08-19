import os
import glob
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List

# ========= CONFIG =========
CSV_DIR   = "csvs/cross_temp"    # parent folder with subfolders of CSVs
TIME_COL  = "Time Elapsed (hours)"
BATT_COL  = "batt_mV"            # open-circuit or near-rest voltage column (VOC)
DEGREE    = 2                    # polynomial degree for SoC(V); will fall back to 1 if needed

# Fault columns to consider (same as your plotting script)
FAULT_COLS = [
    "fault_brownout",
    "fault_sound",
    "fault_sound_brownout",
    "fault_bolt",
    "fault_bolt_brownout",
]

# ======== HELPERS ========

def first_fault_label(df: pd.DataFrame) -> Optional[int]:
    """
    Returns the index label of the first fault, after masking out the first 10% of rows.
    Returns None if no fault found or no fault columns present.
    """
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


def soc_series_from_time(df: pd.DataFrame, idx_fault) -> Optional[pd.Series]:
    """
    Create a SoC (0..1) series for rows from the start up to first fault (inclusive),
    assuming uniform drain over elapsed time.
    SoC = 1 at start, SoC = 0 at first fault.
    """
    if TIME_COL not in df.columns:
        return None

    # Slice from start until the first fault row (inclusive)
    try:
        pos_fault = df.index.get_indexer([idx_fault])[0]
    except Exception:
        return None
    if pos_fault < 0:
        return None

    seg = df.iloc[:pos_fault + 1]
    t = seg[TIME_COL].astype(float)
    if t.isna().any():
        t = t.fillna(method="ffill").fillna(method="bfill")

    t0 = t.iloc[0]
    tf = t.iloc[-1]
    dt = tf - t0
    if not np.isfinite(dt) or dt <= 0:
        return None

    # Uniform drain mapping
    soc = 1.0 - (t - t0) / dt
    soc = soc.clip(lower=0.0, upper=1.0)
    soc.index = seg.index
    return soc


def collect_subfolder_samples(folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a subfolder, aggregate (VOC, SoC) samples from all CSVs.
    Returns arrays x = VOC (float), y = SoC (0..1), possibly empty if nothing usable.
    """
    x_list: List[float] = []
    y_list: List[float] = []

    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if BATT_COL not in df.columns or TIME_COL not in df.columns:
            continue

        idx_fault = first_fault_label(df)
        if idx_fault is None:
            continue

        soc_series = soc_series_from_time(df, idx_fault)
        if soc_series is None:
            continue

        seg = df.loc[soc_series.index]
        # Convert battery column to float
        try:
            voc = pd.to_numeric(seg[BATT_COL], errors="coerce").astype(float)
        except Exception:
            continue

        # Keep rows with finite entries
        mask = np.isfinite(voc.values) & np.isfinite(soc_series.values)
        x_list.extend(voc.values[mask].tolist())
        y_list.extend(soc_series.values[mask].tolist())

    return np.array(x_list, dtype=float), np.array(y_list, dtype=float)


def fit_soc_of_voltage(voc: np.ndarray, soc: np.ndarray, degree: int) -> Optional[np.poly1d]:
    """
    Fit SoC = f(VOC) using polynomial least squares.
    Falls back to linear if insufficient points for the desired degree.
    Returns np.poly1d or None if fitting fails.
    """
    # Need at least degree+1 unique points
    unique_points = len(np.unique(voc[np.isfinite(voc)]))
    deg = degree if unique_points >= (degree + 1) else 1
    if unique_points < 2:
        return None

    try:
        coeffs = np.polyfit(voc, soc, deg=deg)
        return np.poly1d(coeffs)
    except Exception:
        return None


def pretty_poly_str(p: np.poly1d, var: str = "V") -> str:
    """Human-readable polynomial string, e.g., a*V^2 + b*V + c"""
    terms = []
    deg = p.order
    for i, c in enumerate(p.c):
        power = deg - i
        if abs(c) < 1e-12:
            continue
        if power == 0:
            terms.append(f"{c:+.6g}")
        elif power == 1:
            terms.append(f"{c:+.6g}·{var}")
        else:
            terms.append(f"{c:+.6g}·{var}^{power}")
    s = " ".join(terms).lstrip("+").replace("+ -", "- ")
    return s if s else "0"


def invert_soc_function(p: np.poly1d, soc_target: float, v_min: float, v_max: float) -> float | None:
    """
    Solve p(V) = soc_target for V. Picks a real root within [v_min, v_max].
    If multiple roots in range, pick the one closest to midpoint of the range.
    """
    # Adjust constant term on a raw coeff array, then re-wrap
    coeffs = np.array(p.coeffs, dtype=float)
    coeffs[-1] -= soc_target
    q = np.poly1d(coeffs)

    # Exact roots
    roots = q.r
    real_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-7]
    in_range = [r for r in real_roots if (v_min - 1e-6) <= r <= (v_max + 1e-6)]
    if in_range:
        mid = 0.5 * (v_min + v_max)
        return min(in_range, key=lambda r: abs(r - mid))

    # Fallback: coarse numeric search over observed range
    vs = np.linspace(v_min, v_max, 2000)
    ys = p(vs) - soc_target
    try:
        idx = int(np.argmin(np.abs(ys)))
        return float(vs[idx])
    except Exception:
        return None



# ========== MAIN ==========

def main():
    if not os.path.isdir(CSV_DIR):
        print(f"Missing directory: {CSV_DIR}")
        return

    subfolders = sorted(
        d for d in os.listdir(CSV_DIR)
        if os.path.isdir(os.path.join(CSV_DIR, d))
    )
    if not subfolders:
        print(f"No subfolders found in {CSV_DIR}")
        return

    print(f"Scanning {len(subfolders)} subfolders in {CSV_DIR}...\n")
    print("RESULTS")
    print("=======")

    for sf in subfolders:
        folder_path = os.path.join(CSV_DIR, sf)
        voc, soc = collect_subfolder_samples(folder_path)

        if len(voc) == 0 or len(soc) == 0:
            print(f"\n[{sf}] No usable data (no faults found or missing columns).")
            continue

        # Fit SoC(V)
        p = fit_soc_of_voltage(voc, soc, DEGREE)
        if p is None:
            print(f"\n[{sf}] Could not fit SoC(V) (insufficient or invalid data).")
            continue
        import matplotlib.pyplot as plt

        
        v_min, v_max = float(np.min(voc)), float(np.max(voc))
        func_str = pretty_poly_str(p, var="V")

        # Plot SoC(V) fit and data points
        plt.figure(figsize=(7, 4))
        plt.scatter(voc, soc, s=8, alpha=0.5, label="Samples")
        v_plot = np.linspace(v_min, v_max, 300)
        plt.plot(v_plot, p(v_plot), "r-", lw=2, label="Fitted SoC(V)")
        plt.xlabel("VOC (mV)")
        plt.ylabel("SoC (0..1)")
        plt.title(f"SoC(V) fit for {sf}")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Invert: find voltage at 10% SoC and 1% SoC
        v_at_10 = invert_soc_function(p, soc_target=0.10, v_min=v_min, v_max=v_max)
        v_at_01 = invert_soc_function(p, soc_target=0.01, v_min=v_min, v_max=v_max)

        # Print
        print(f"\n[{sf}]")
        print(f"  Samples used: {len(voc)}")
        print(f"  Voltage range (mV): {v_min:.2f} .. {v_max:.2f}")
        print(f"  Fitted SoC(V)  (SoC in 0..1):  SoC(V) = {func_str}")
        if v_at_10 is not None:
            print(f"  Inverse: V@10% SoC  ≈ {v_at_10:.2f} mV")
        else:
            print(f"  Inverse: V@10% SoC  ≈ (no real root in range)")
        if v_at_01 is not None:
            print(f"  Inverse: V@1%  SoC  ≈ {v_at_01:.2f} mV")
        else:
            print(f"  Inverse: V@1%  SoC  ≈ (no real root in range)")

if __name__ == "__main__":
    main()
