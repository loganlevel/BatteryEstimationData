import sys
import pandas as pd
import matplotlib.pyplot as plt
import re
from io import StringIO
import os
from datetime import datetime
import numpy as np

# Configurable flag for low-pass filtering
APPLY_LOW_PASS_FILTER = True  # Set to False to disable filtering


def main():
    print("Paste the CSV (with columns: timestamp,batt_mV), then press Enter then Ctrl+D (macOS/Linux) or Ctrl+Z then Enter (Windows):\n")
    raw_input = sys.stdin.read()

    csv_match = re.search(r"(timestamp\s*,\s*batt_mV.*?)\n([\s\S]+)", raw_input, re.IGNORECASE)
    if not csv_match:
        print("Failed to find CSV with header 'timestamp,batt_mV'.")
        return

    csv_header = csv_match.group(1).strip()
    csv_data = csv_match.group(2).strip()

    df = pd.read_csv(StringIO(csv_header + "\n" + csv_data))
    df.columns = [c.strip() for c in df.columns]
    if set(map(str.lower, df.columns)) != {"timestamp", "batt_mv"}:
        print("CSV must contain exactly 'timestamp' and 'batt_mV' columns.")
        return

    df = df.sort_values("timestamp").reset_index(drop=True)
    start_ts = df["timestamp"].iloc[0]
    df["Time Elapsed (hours)"] = (df["timestamp"] - start_ts) / 3600.0

    date_str = datetime.now().strftime("%Y%m%d")
    duration_h = df["Time Elapsed (hours)"].iloc[-1] if len(df) > 1 else 0.0
    vmin = int(df["batt_mV"].min())
    vmax = int(df["batt_mV"].max())
    filename = f"batt_{date_str}_{duration_h:.2f}h_{vmin}-{vmax}mV"

    os.makedirs("csvs/parsed", exist_ok=True)
    parsed_csv_path = f"csvs/parsed/{filename}.csv"
    df.to_csv(parsed_csv_path, index=False)
    print(f"Saved parsed CSV to: {parsed_csv_path}")

    os.makedirs(f"plots/{filename}", exist_ok=True)
    out_csv_copy = f"plots/{filename}/{filename}.csv"
    df.to_csv(out_csv_copy, index=False)
    print(f"Saved parsed CSV copy to: {out_csv_copy}")

    if APPLY_LOW_PASS_FILTER:
        RELAXATION_LOW_PASS_FILTER_ALPHA = 0.2  # Adjust alpha as needed (0 < alpha < 1)
        filtered = []
        prev = df["batt_mV"].iloc[0]
        for v in df["batt_mV"]:
            filtered_v = RELAXATION_LOW_PASS_FILTER_ALPHA * v + (1.0 - RELAXATION_LOW_PASS_FILTER_ALPHA) * prev
            filtered.append(filtered_v)
            prev = filtered_v
        df["batt_mV_filtered"] = filtered
        voltage_col = "batt_mV_filtered"
        plot_label = "Filtered Voltage"
        plot_title = "Battery Voltage vs Time (Low-pass Filtered)"
    else:
        voltage_col = "batt_mV"
        plot_label = "Raw Voltage"
        plot_title = "Battery Voltage vs Time (Raw)"

    plt.figure(figsize=(12, 5))
    df["Time Elapsed (minutes)"] = (df["timestamp"] - start_ts) / 60.0
    plt.plot(df["Time Elapsed (minutes)"], df[voltage_col], label=plot_label)
    plt.title(plot_title)
    plt.xlabel("Time Elapsed (minutes)")
    plt.ylabel("Voltage (mV)")
    plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
    plt.minorticks_on()

    min_minute = int(np.floor(df["Time Elapsed (minutes)"].min() / 15) * 15)
    max_minute = int(np.ceil(df["Time Elapsed (minutes)"].max() / 15) * 15)
    plt.xticks(np.arange(min_minute, max_minute + 1, 15))

    try:
        ymin = max(0, df[voltage_col].min() - 50)
        ymax = df[voltage_col].max() + 50
        plt.ylim(ymin, ymax)
    except Exception:
        pass

    out_png = f"plots/{filename}/{filename}_voltage.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to: {out_png}")

    if len(df) > 1:
        print(f"Final duration: {duration_h:.2f} hours")

if __name__ == "__main__":
    main()
