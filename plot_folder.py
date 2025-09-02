import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np  # NEW
from matplotlib.lines import Line2D

# CONFIGURATION
CSV_DIR = "csvs/manufacturers-cross-temp/temp-60c/level"
CSV_TAG = os.path.basename(os.path.normpath(CSV_DIR))
PLOT_OUTPUT = f"plots/compare_all_{CSV_DIR.replace("/", "-")}.png"
COMPUTE_LOW_BATT_BY_ALL_FAULTS=True
TIME_LIMIT_HOURS=80

COLUMNS_TO_PLOT = [
    ("temp", "Temperature", "°C"),
    ("batt_mV", "Battery Voltage", "mV"),
    ("soundDroop_mV", "Silent Tone Min Voltage", "mV"),
    ("boltDroop_mV", "Bolt Op Min Voltage", "mV"),
    ("soundDroopMag_mV", "Silent Tone Droop", "mV"),
    ("boltDroopMag_mV", "Bolt Op Droop", "mV"),
]

# Find CSVs
csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
if not csv_files:
    print(f"No CSV files found in {CSV_DIR}")
    exit(1)

print(f"Found {len(csv_files)} CSVs to compare in {CSV_DIR}")

# Prepare plot
fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
axes = axes.flatten()

dot_size = 20

# Collect series line handles (for the legend) only from subplot[0]
series_handles_axis0 = []

# Track which fault columns exist across all CSVs (for proxy legend)
fault_present = {
    "fault_brownout": False,
    "fault_sound": False,
    "fault_sound_brownout": False,
    "fault_bolt": False,
    "fault_bolt_brownout": False,
}

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    df = df.iloc[:(TIME_LIMIT_HOURS*2)]

    if "Time Elapsed (hours)" not in df.columns:
        print(f"Skipping {csv_path} (missing time column)")
        continue

    label = os.path.basename(csv_path).replace(".csv", "").split("_")[2]
    t_series = df["Time Elapsed (hours)"]
    total_time = t_series.iloc[-1] - t_series.iloc[0]
    offset = total_time * 0.005

    # Update which fault types appear in any file
    for k in fault_present.keys():
        if k in df.columns and df[k].any():
            fault_present[k] = True

    # Compute 10% pre-first-fault "low battery" index (per-file)
    if COMPUTE_LOW_BATT_BY_ALL_FAULTS:
        fault_cols = [c for c in ["fault_brownout", "fault_sound", "fault_sound_brownout", "fault_bolt", "fault_bolt_brownout"] if c in df.columns]
    else:
        fault_cols = [c for c in ["fault_bolt", "fault_bolt_brownout"] if c in df.columns]
    idx_low_batt = None
    if fault_cols:
        any_fault = df[fault_cols].any(axis=1)
        if any_fault.any():
            # Ignore faults in the first 10% of the data
            n_rows = len(df)
            min_idx = int(np.ceil(0.10 * n_rows))
            any_fault.iloc[:min_idx] = False
            if any_fault.any():
                idx_first_fault = any_fault.idxmax()
                t0 = t_series.iloc[0]
                t_fault = t_series.loc[idx_first_fault]
                hours_to_fault = t_fault - t0
                t_target = t_fault - 0.10 * hours_to_fault
                idx_low_batt = (t_series - t_target).abs().idxmin()
                hours_to_fault = t_fault - t0
                t_target = t_fault - 0.10 * hours_to_fault
                idx_low_batt = (t_series - t_target).abs().idxmin()

    for i, (col, title, unit) in enumerate(COLUMNS_TO_PLOT):
        ax = axes[i]
        if col not in df.columns:
            continue

        # Plot series; capture handle on the first subplot only
        line, = ax.plot(t_series, df[col], label=label)
        if i == 0:
            series_handles_axis0.append(line)

        # Fault markers — same offsets/order as your single-file script
        if "fault_brownout" in df.columns:
            ax.scatter(t_series[df["fault_brownout"]] - offset,
                       df[col][df["fault_brownout"]],
                       color="red", s=dot_size, zorder=5, label=None)

        # Sound first (+offsets)
        if "fault_sound" in df.columns:
            ax.scatter(t_series[df["fault_sound"]] + offset,
                       df[col][df["fault_sound"]],
                       color="purple", s=dot_size, zorder=5, label=None)
        if "fault_sound_brownout" in df.columns:
            ax.scatter(t_series[df["fault_sound_brownout"]] + 2*offset,
                       df[col][df["fault_sound_brownout"]],
                       color="blue", s=dot_size, zorder=5, label=None)

        # Then bolt (-offsets)
        if "fault_bolt" in df.columns:
            ax.scatter(t_series[df["fault_bolt"]] - offset,
                       df[col][df["fault_bolt"]],
                       color="orange", s=dot_size, zorder=5, label=None)
        if "fault_bolt_brownout" in df.columns:
            ax.scatter(t_series[df["fault_bolt_brownout"]] - 2*offset,
                       df[col][df["fault_bolt_brownout"]],
                       color="green", s=dot_size, zorder=5, label=None)

        # Low-battery crosshairs (not part of series legend)
        if idx_low_batt is not None and 0 <= idx_low_batt < len(df):
            lc = line.get_color()
            ax.axvline(t_series.loc[idx_low_batt], linestyle="dotted", linewidth=1.5, color=lc, zorder=6)
            ax.axhline(df[col].loc[idx_low_batt], linestyle="dotted", linewidth=1.5, color=lc, zorder=6)

        ax.set_title(f"{title} vs Time Elapsed")
        ax.set_ylabel(unit)
        ax.grid(True)

        # y-lims
        if col == "temp":
            ax.set_ylim(-30, 70)
        elif col == "batt_mV":
            ax.set_ylim(2000, 3400)
        elif col in ("soundDroopMag_mV", "boltDroopMag_mV"):
            ax.set_ylim(0, 2500)
        else:
            ax.set_ylim(0, 3400)

# ---- Legends: only on the first subplot, after all data is plotted ----
ax0 = axes[0]

# Series legend (lines only; excludes low-battery crosshairs)
series_legend = ax0.legend(handles=series_handles_axis0, title="Series", loc="lower right")
ax0.add_artist(series_legend)

# Faults legend (proxy handles + Low battery)
fault_handles = []
if fault_present["fault_brownout"]:
    fault_handles.append(Line2D([], [], color="red", marker='o', linestyle='None', markersize=dot_size/5, label="Brownout"))
if fault_present["fault_sound"]:
    fault_handles.append(Line2D([], [], color="purple", marker='o', linestyle='None', markersize=dot_size/5, label="Silent Tone Op Fail"))
if fault_present["fault_sound_brownout"]:
    fault_handles.append(Line2D([], [], color="blue", marker='o', linestyle='None', markersize=dot_size/5, label="Silent Tone Brownout"))
if fault_present["fault_bolt"]:
    fault_handles.append(Line2D([], [], color="orange", marker='o', linestyle='None', markersize=dot_size/5, label="Bolt Op Fail"))
if fault_present["fault_bolt_brownout"]:
    fault_handles.append(Line2D([], [], color="green", marker='o', linestyle='None', markersize=dot_size/5, label="Bolt Brownout"))
fault_handles.append(Line2D([], [], color="black", linestyle="dotted", linewidth=1.5, label="Low battery"))

fault_legend = ax0.legend(handles=fault_handles, title="Faults", loc="lower left")
ax0.add_artist(fault_legend)

# X-axis label
for ax in axes:
    ax.set_xlabel("Time Elapsed (hours)")

# Save figure
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig(PLOT_OUTPUT)
print(f"Saved comparison plot to {PLOT_OUTPUT}")
