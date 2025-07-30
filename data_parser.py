import sys
import pandas as pd
import matplotlib.pyplot as plt
import re
from io import StringIO
import os
from datetime import datetime

def main():
    print("Paste the full input below, then press Enter then Ctrl+D (Linux/macOS) or Ctrl+Z then Enter (Windows):\n")
    raw_input = sys.stdin.read()

    # Extract metadata
    def extract(key, fallback="UNKNOWN"):
        match = re.search(rf"{key}:\s+(.+)", raw_input)
        return match.group(1).strip() if match else fallback

    product_name = extract("PRODUCT_NAME")
    hw_rev = extract("HW_REV")
    chip_id = extract("CHIP_ID")
    build_time_str = extract("FW_BUILD_TIME", "")
    droop_volume = extract("droopVolume")
    load_volume = extract("loadVolume")
    load_dur = extract("loadDurationSeconds")
    relax_dur = extract("relaxDurationSeconds")
    bolt_checks = extract("boltChecks")
    try:
        build_dt = datetime.strptime(build_time_str, "%b %d %Y %H:%M:%S")
        date_str = build_dt.strftime("%Y%m%d")
    except ValueError:
        date_str = datetime.now().strftime("%Y%m%d")

    # Extract CSV data
    csv_match = re.search(r"(timestamp,.*?)\n([\s\S]+)", raw_input)
    if not csv_match:
        print("Failed to find CSV data section.")
        return

    csv_header = csv_match.group(1).strip()
    csv_data = csv_match.group(2).strip()
    df = pd.read_csv(StringIO(csv_header + '\n' + csv_data))
    df.columns = df.columns.str.strip()

    # Compute derived fields
    start_ts = df["timestamp"].iloc[0]
    end_ts = df["timestamp"].iloc[-1]
    df["Time Elapsed (hours)"] = (df["timestamp"] - start_ts) / 3600
    df["soundDroopMag_mV"] = df["batt_mV"] - df["soundDroop_mV"]
    df["boltDroopMag_mV"] = df["batt_mV"] - df["boltDroop_mV"]

    filename = f"{product_name}_{hw_rev}_{chip_id}_{date_str}_droop{droop_volume}_load{load_volume}_loaddur{load_dur}_relaxdur{relax_dur}_boltchecks{bolt_checks}"


    # Save processed CSV
    os.makedirs("csvs", exist_ok=True)
    output_csv = f"csvs/{filename}.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved parsed CSV to: {output_csv}")

    # Prepare plotting
    os.makedirs("plots", exist_ok=True)
    brownout_mask = df["brownout"] == True
    time_col = "Time Elapsed (hours)"

    plots = [
        ("temp", "Temperature", "°C"),
        ("batt_mV", "Battery Voltage", "mV"),
        ("soundDroop_mV", "Silent Tone Min Voltage", "mV"),
        ("boltDroop_mV", "Bolt Op Min Voltage", "mV"),
        ("soundDroopMag_mV", "Silent Tone Droop", "mV"),
        ("boltDroopMag_mV", "Bolt Op Droop", "mV"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for i, (col, param, unit) in enumerate(plots):
        ax = axes[i]
        ax.plot(df[time_col], df[col], label=param)
        ax.scatter(df[time_col][brownout_mask], df[col][brownout_mask],
                   color="red", label="Brownout Event", zorder=5)
        ax.set_title(f"{param} vs Time Elapsed")
        ax.set_ylabel(unit)
        ax.grid(True)
        ax.legend()

    for ax in axes:
        ax.set_xlabel("Time Elapsed (hours)")

    plt.tight_layout()
    output_png = f"plots/{filename}_all_plots.png"
    plt.savefig(output_png)
    print(f"Saved all plots as: {output_png}")

if __name__ == "__main__":
    main()
