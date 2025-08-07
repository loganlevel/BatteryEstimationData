import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 csv_data_parser.py <csv_file_path>")
        return

    csv_file_path = sys.argv[1]

    if not os.path.exists(csv_file_path):
        print(f"CSV file '{csv_file_path}' not found.")
        return

    # Derive base name for output files from CSV file name (no extension)
    filename = os.path.splitext(os.path.basename(csv_file_path))[0]

    # Load the CSV file
    df = pd.read_csv(csv_file_path)
    df.columns = df.columns.str.strip()

    if "Timestamp" not in df.columns:
        print("Missing 'Timestamp' column in CSV.")
        return

    # Add computed columns
    start_ts = df["Timestamp"].iloc[0]
    df["Time Elapsed (hours)"] = (df["Timestamp"] - start_ts) / 3600
    if "Battery Voltage (mV)" in df.columns and "Min Droop Voltage (mV)" in df.columns:
        df["Droop (mV)"] = df["Battery Voltage (mV)"] - df["Min Droop Voltage (mV)"]

    # Save modified CSV
    os.makedirs("csvs", exist_ok=True)
    output_csv = os.path.join("csvs", f"{filename}.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved parsed CSV to: {output_csv}")

    # Determine which columns to plot
    y_columns = [col for col in df.columns if col not in ["Timestamp", "Time Elapsed (hours)"]]
    num_plots = len(y_columns)

    # Determine subplot grid size
    cols = 2
    rows = math.ceil(num_plots / cols)

    # Create subplots
    fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axs = axs.flatten()

    for i, col in enumerate(y_columns):
        axs[i].plot(df["Time Elapsed (hours)"], df[col])
        axs[i].set_title(f"{col} vs Time")
        axs[i].set_xlabel("Time Elapsed (hours)")
        axs[i].set_ylabel(col)
        axs[i].grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    output_plot = os.path.join("plots", f"{filename}_summary.png")
    plt.savefig(output_plot)
    print(f"Saved all plots to: {output_plot}")

if __name__ == "__main__":
    main()
