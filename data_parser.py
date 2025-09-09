import sys
import pandas as pd
import matplotlib.pyplot as plt
import re
from io import StringIO
import os
from datetime import datetime

dut_names_dict = {
    "0xf820357251287ce1": "BE-1",
    "0x07540ba7e21004bf": "BE-2",
    "0x679f80f90cea91dd": "BE-3",
    "0x77db265172dec07f": "BE-4",
    "0xd0213d8230e21d84": "BE-5",
    "0x862b09ac4bbb4604": "BE-6",
    "0xb14c93f83f6fa826": "BE-7",
    "0x773051536a181f4a": "BE-8",
    "0x9c15f93d401b4598": "BE-8",
    "0x5c84a4c09ecf87d8": "BE-9",
    "0xb014cb8c55f6928a": "BE-10",
    "0xe35fb8691e3cec33": "BE-11",
    "0xc9b118b3f0383962": "BE-12",
    "0x178cddb9c4876c64": "BE-13",
    "0x839c39e2f387076a": "BE-14",
    "0x32b7e0c75a5f3d62": "BE-19",
}

def main():
    print("Paste the full input below, then press Enter then Ctrl+D (Linux/macOS) or Ctrl+Z then Enter (Windows):\n")
    raw_input = sys.stdin.read()

    def extract(key, fallback="UNKNOWN"):
        match = re.search(rf"{key}:\s+(.+)", raw_input)
        return match.group(1).strip() if match else fallback

    product_name = extract("PRODUCT_NAME")
    hw_rev = extract("HW_REV")
    dut_id = dut_names_dict[extract("CHIP_ID")]
    build_time_str = extract("FW_BUILD_TIME", "")
    droop_volume = extract("droopVolume")
    load_volume = extract("loadVolume")
    load_dur = extract("loadDurationSeconds")
    relax_dur = extract("relaxDurationSeconds")
    bolt_checks = extract("boltChecks")


    build_dt = datetime.strptime(build_time_str, "%b %d %Y %H:%M:%S")
    build_dt = build_dt.strftime("%Y%m%d")

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
    # Make sure batteryLevel is an integer dtype
    # df["batteryLevel"] = df["batteryLevel"].astype(int)

    # Decode batteryLevel: lower 4 bits = meanLevel, upper 4 bits = nnLevel
    df["meanLevel"] = df["batteryLevel"] & 0x0F
    df["nnLevel"]   = (df["batteryLevel"] // 16) & 0x0F


    # Decode fault bitfields
    df["fault_brownout"] = (df["faults"] & (1 << 0)) > 0
    df["fault_bolt"] = (df["faults"] & (1 << 1)) > 0
    df["fault_sound"] = (df["faults"] & (1 << 2)) > 0
    df["fault_sound_brownout"] = (df["faults"] & (1 << 3)) > 0
    df["fault_bolt_brownout"] = (df["faults"] & (1 << 4)) > 0

    filename = f"{product_name}_{hw_rev}_{dut_id}_{build_dt}-{date_str}_droop{droop_volume}_load{load_volume}_loaddur{load_dur}_relaxdur{relax_dur}_boltchecks{bolt_checks}"

    # Save processed CSV
    os.makedirs("csvs/parsed", exist_ok=True)
    output_csv = f"csvs/parsed/{filename}.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved parsed CSV to: {output_csv}")

    os.makedirs(f"plots/{filename}", exist_ok=True)
    output_csv = f"plots/{filename}/{filename}.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved parsed CSV to: {output_csv}")

    # Prepare plotting
    time_col = "Time Elapsed (hours)"
    os.makedirs("plots", exist_ok=True)

    # If boltDroop_mV is all zeros, set boltDroopMag_mV to all zeros
    if (df["boltDroop_mV"] == 0).all():
        df["boltDroopMag_mV"] = 0
        df.to_csv(output_csv, index=False)

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

    total_time = df[time_col].iloc[-1] - df[time_col].iloc[0]
    offset = total_time * 0.005  # 0.5% of the total duration
    dot_size = 20

    for i, (col, param, unit) in enumerate(plots):
        ax = axes[i]
        ax.plot(df[time_col], df[col], label=param)

        # Add fault annotations with slight horizontal offsets
        ax.scatter(df[time_col][df["fault_brownout"]] - offset,
               df[col][df["fault_brownout"]],
               color="red", label="Brownout", zorder=5, s=dot_size)

        # Plot sound faults first
        ax.scatter(df[time_col][df["fault_sound"]] + offset,
               df[col][df["fault_sound"]],
               color="purple", label="Sound Op Fail", zorder=5, s=dot_size)

        ax.scatter(df[time_col][df["fault_sound_brownout"]] + 2*offset,
               df[col][df["fault_sound_brownout"]],
               color="blue", label="Sound Brownout", zorder=5, s=dot_size)

        # Then plot bolt faults
        ax.scatter(df[time_col][df["fault_bolt"]] - offset,
               df[col][df["fault_bolt"]],
               color="orange", label="Bolt Op Fail", zorder=5, s=dot_size)

        ax.scatter(df[time_col][df["fault_bolt_brownout"]] - 2*offset,
               df[col][df["fault_bolt_brownout"]],
               color="green", label="Bolt Brownout", zorder=5, s=dot_size)

        ax.set_title(f"{param} vs Time Elapsed")
        ax.set_ylabel(unit)
        ax.grid(True)
        ax.legend()

        # Set y-axis limits
        if col == "temp":
            ax.set_ylim(-30, 70)
        else:
            ax.set_ylim(0, 3400)  # mV for voltage plots

    for ax in axes:
        ax.set_xlabel("Time Elapsed (hours)")

    plt.tight_layout()
    output_png = f"plots/{filename}/{filename}_all_plots.png"
    plt.savefig(output_png)
    print(f"Saved all plots as: {output_png}")

    # Print duration up to the first fault
    fault_mask = (
        df["fault_brownout"] |
        df["fault_bolt"] |
        df["fault_sound"] |
        df["fault_sound_brownout"] |
        df["fault_bolt_brownout"]
    )
    if fault_mask.any():
        # Ignore the first 10% of data when searching for the first fault
        ignore_n = int(len(df) * 0.1)
        fault_mask_ignored = fault_mask.copy()
        fault_mask_ignored.iloc[:ignore_n] = False
        if fault_mask_ignored.any():
            first_fault_idx = fault_mask_ignored.idxmax()
        else:
            first_fault_idx = len(df) - 1  # No fault after ignoring, use last index
        duration_to_first_fault = df[time_col].iloc[first_fault_idx] - df[time_col].iloc[0]
        print(f"Duration to first fault: {duration_to_first_fault:.2f} hours")
    else:
        final_duration = df[time_col].iloc[-1] - df[time_col].iloc[0]
        print(f"Final test duration: {final_duration:.2f} hours")

if __name__ == "__main__":
    main()
