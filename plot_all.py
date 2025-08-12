import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# CONFIGURATION
CSV_DIR = "csvs/temp-20"
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
    total_time = df["Time Elapsed (hours)"].iloc[-1] - df["Time Elapsed (hours)"].iloc[0]
    offset = total_time * 0.005

    for i, (col, title, unit) in enumerate(COLUMNS_TO_PLOT):
        ax = axes[i]
        if col not in df.columns:
            continue

        # Main line plot (with label for the series legend)
        ax.plot(df["Time Elapsed (hours)"], df[col], label=label)

        # Only create dummy handles for fault legend once (first CSV only)
        if csv_path == csv_files[0]:
            fault_handles = []
            if "fault_brownout" in df.columns:
                fault_handles.append(plt.Line2D([], [], color="red", marker='o', linestyle='None', markersize=dot_size/5, label="Brownout"))
            if "fault_bolt" in df.columns:
                fault_handles.append(plt.Line2D([], [], color="orange", marker='o', linestyle='None', markersize=dot_size/5, label="Bolt Fail"))
            if "fault_sound" in df.columns:
                fault_handles.append(plt.Line2D([], [], color="purple", marker='o', linestyle='None', markersize=dot_size/5, label="Sound Fail"))

        # Plot actual fault markers (no label so they don't go in the series legend)
        if "fault_brownout" in df.columns:
            ax.scatter(df["Time Elapsed (hours)"][df["fault_brownout"]] - offset,
                    df[col][df["fault_brownout"]],
                    color="red", s=dot_size, zorder=5, label=None)

        if "fault_bolt" in df.columns:
            ax.scatter(df["Time Elapsed (hours)"][df["fault_bolt"]],
                    df[col][df["fault_bolt"]],
                    color="orange", s=dot_size, zorder=5, label=None)

        if "fault_sound" in df.columns:
            ax.scatter(df["Time Elapsed (hours)"][df["fault_sound"]] + offset,
                    df[col][df["fault_sound"]],
                    color="purple", s=dot_size, zorder=5, label=None)

        ax.set_title(f"{title} vs Time Elapsed")
        ax.set_ylabel(unit)
        ax.grid(True)

        if i == 0:
            # First legend: series lines only
            series_legend = ax.legend(title="Series", loc="upper left")
            ax.add_artist(series_legend)

            # Second legend: faults only
            ax.legend(handles=fault_handles, title="Faults", loc="upper right")

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
