import sys
import pandas as pd
import matplotlib.pyplot as plt
import re
from io import StringIO
import os

'''
Changes to make:
* Use current date and elapsed time to to add the date range to the file name
* Concatonate all the plots into a single png and then just save everything in the same folde
'''

def main():
    print("Paste the full input below, then press Enter then Ctrl+D (Linux/macOS) or Ctrl+Z then Enter (Windows):\n")

    raw_input = sys.stdin.read()

    # Separate metadata from data
    data_match = re.search(r"(Timestamp,.*?)\n([\s\S]+)", raw_input)
    if not data_match:
        print("Failed to find CSV data section.")
        return

    csv_header = data_match.group(1).strip()
    csv_data = data_match.group(2).strip()
    df = pd.read_csv(StringIO(csv_header + '\n' + csv_data))
    df.columns = df.columns.str.strip()


    # Add computed columns
    start_ts = df["Timestamp"].iloc[0]
    df["Time Elapsed (hours)"] = (df["Timestamp"] - start_ts) / 3600
    df["Droop (mV)"] = df["Battery Voltage (mV)"] - df["Min Droop Voltage (mV)"]


    # Parse metadata
    def extract(key, fallback="UNKNOWN"):
        match = re.search(rf"{key}:\s+(.+)", raw_input)
        return match.group(1).strip() if match else fallback

    product_name = extract("PRODUCT_NAME")
    hw_rev = extract("HW_REV")
    load_power = extract("loadPowerLevel")
    period = extract("loadPeriodSeconds")
    duty = extract("loadDutyCycle")
    chip_id = extract("CHIP_ID")

    filename = f"{product_name}_{hw_rev}_Power-{load_power}_Period-{period}_Duty-{duty}_{chip_id}"
    output_csv = f"csvs/{filename}.csv"

    # Add computed columns
    start_ts = df["Timestamp"].iloc[0]
    df["Time Elapsed (hours)"] = (df["Timestamp"] - start_ts) / 3600
    df["Droop (mV)"] = df["Battery Voltage (mV)"] - df["Min Droop Voltage (mV)"]

    # Save modified CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved parsed CSV to: {output_csv}")

    # Plotting
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    y_columns = [col for col in df.columns if col not in ["Timestamp", "Time Elapsed (hours)"]]
    n_plots = len(y_columns)
    n_rows, n_cols = 2, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    for i, col in enumerate(y_columns):
        if i >= n_rows * n_cols:
            break
        ax = axes[i]
        ax.plot(df["Time Elapsed (hours)"], df[col])
        ax.set_ylabel(col)
        ax.set_title(f"{col} vs Time Elapsed")
        ax.grid(True)

    for ax in axes:
        ax.set_xlabel("Time Elapsed (hours)")

    plt.tight_layout()
    output_png = f"{plot_dir}/{filename}_all_plots.png"
    plt.savefig(output_png)
    print(f"Saved all plots as: {output_png}")

if __name__ == "__main__":
    main()
