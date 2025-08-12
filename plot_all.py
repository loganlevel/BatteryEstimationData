import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np  # NEW

# CONFIGURATION
CSV_DIR = "csvs/temp-60"
CSV_TAG = os.path.basename(os.path.normpath(CSV_DIR))
PLOT_OUTPUT = f"plots/compare_all_{CSV_TAG}.png"

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

# Offset and dot styling
dot_size = 20

for csv_path in csv_files:
    df = pd.read_csv(csv_path)

    if "Time Elapsed (hours)" not in df.columns:
        print(f"Skipping {csv_path} (missing time column)")
        continue

    label = os.path.basename(csv_path).replace(".csv", "").split("_")[2]
    t_series = df["Time Elapsed (hours)"]
    total_time = t_series.iloc[-1] - t_series.iloc[0]
    offset = total_time * 0.005

    # --- NEW: compute index for "Low battery" (10% before first fault) ---
    fault_cols = [c for c in ["fault_brownout", "fault_bolt", "fault_sound"] if c in df.columns]
    idx_low_batt = None
    if fault_cols:
        any_fault = df[fault_cols].any(axis=1)
        if any_fault.any():
            idx_first_fault = any_fault.idxmax()  # first True
            t0 = t_series.iloc[0]
            t_fault = t_series.loc[idx_first_fault]
            hours_to_fault = t_fault - t0
            t_target = t_fault - 0.10 * hours_to_fault
            idx_low_batt = (t_series - t_target).abs().idxmin()
    # ---------------------------------------------------------------------

    for i, (col, title, unit) in enumerate(COLUMNS_TO_PLOT):
        ax = axes[i]
        if col not in df.columns:
            continue

        # Main line plot (with label for the series legend)
        ax.plot(t_series, df[col], label=label)

        # Only create dummy handles for fault legend once (first CSV only)
        if csv_path == csv_files[0]:
            fault_handles = []
            if "fault_brownout" in df.columns:
                fault_handles.append(plt.Line2D([], [], color="red", marker='o', linestyle='None', markersize=dot_size/5, label="Brownout"))
            if "fault_bolt" in df.columns:
                fault_handles.append(plt.Line2D([], [], color="orange", marker='o', linestyle='None', markersize=dot_size/5, label="Bolt Fail"))
            if "fault_sound" in df.columns:
                fault_handles.append(plt.Line2D([], [], color="purple", marker='o', linestyle='None', markersize=dot_size/5, label="Sound Fail"))
            # NEW: Low battery (10% pre-fault X)
            fault_handles.append(plt.Line2D([], [], color="black", linestyle="dotted", linewidth=1.5, label="Low battery"))

        # Plot actual fault markers (no label so they don't go in the series legend)
        if "fault_brownout" in df.columns:
            ax.scatter(t_series[df["fault_brownout"]] - offset,
                       df[col][df["fault_brownout"]],
                       color="red", s=dot_size, zorder=5, label=None)

        if "fault_bolt" in df.columns:
            ax.scatter(t_series[df["fault_bolt"]],
                       df[col][df["fault_bolt"]],
                       color="orange", s=dot_size, zorder=5, label=None)

        if "fault_sound" in df.columns:
            ax.scatter(t_series[df["fault_sound"]] + offset,
                       df[col][df["fault_sound"]],
                       color="purple", s=dot_size, zorder=5, label=None)

        # NEW: plot "Low battery" X marker for this dataset/column if computed
        if idx_low_batt is not None and 0 <= idx_low_batt < len(df):
            # Get the color of the current series line
            line_color = ax.get_lines()[-1].get_color() if ax.get_lines() else "black"
            # Draw dotted vertical and horizontal lines at the low battery point, matching series color
            ax.axvline(
            x=t_series.loc[idx_low_batt],
            linestyle="dotted",
            linewidth=1.5,
            color=line_color,
            zorder=6
            )
            ax.axhline(
            y=df[col].loc[idx_low_batt],
            linestyle="dotted",
            linewidth=1.5,
            color=line_color,
            zorder=6
            )

        ax.set_title(f"{title} vs Time Elapsed")
        ax.set_ylabel(unit)
        ax.grid(True)

        if i == 0:
            # First legend: series lines only
            # Place series legend in the best location automatically
            series_legend = ax.legend(title="Series", loc="lower right")
            ax.add_artist(series_legend)

            # Place fault legend in another best location, avoiding overlap
            fault_legend = ax.legend(handles=fault_handles, title="Faults", loc="lower left")
            ax.add_artist(fault_legend)

        if col == "temp":
            ax.set_ylim(-30, 70)
        elif col == "batt_mV":
            ax.set_ylim(2000, 3400)
        elif col == "soundDroopMag_mV" or col == "boltDroopMag_mV":
            ax.set_ylim(0, 2500)
        else:
            ax.set_ylim(0, 3400)

# X-axis label
for ax in axes:
    ax.set_xlabel("Time Elapsed (hours)")

# Save figure
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig(PLOT_OUTPUT)
print(f"Saved comparison plot to {PLOT_OUTPUT}")
