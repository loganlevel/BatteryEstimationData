import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

def decode_levels_and_faults(df):
    if df["batteryLevel"].dtype != int and df["batteryLevel"].dtype != "int64":
        df["batteryLevel"] = df["batteryLevel"].astype(int)
    df["meanLevel"] = df["batteryLevel"] & 0x0F
    df["nnLevel"] = (df["batteryLevel"] // 16) & 0x0F
    df["fault_brownout"] = (df["faults"] & (1 << 0)) > 0
    df["fault_bolt"] = (df["faults"] & (1 << 1)) > 0
    df["fault_sound"] = (df["faults"] & (1 << 2)) > 0
    df["fault_sound_brownout"] = (df["faults"] & (1 << 3)) > 0
    df["fault_bolt_brownout"] = (df["faults"] & (1 << 4)) > 0
    return df

def regenerate_png_from_csv(csv_path, png_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    required = {"timestamp","batt_mV","soundDroop_mV","boltDroop_mV","batteryLevel","faults","temp"}
    if not required.issubset(df.columns):
        print(f"[skip] Missing columns in {csv_path}: {sorted(list(required - set(df.columns)))}")
        return
    start_ts = df["timestamp"].iloc[0]
    df["Time Elapsed (hours)"] = (df["timestamp"] - start_ts) / 3600
    df["soundDroopMag_mV"] = df["batt_mV"] - df["soundDroop_mV"]
    df["boltDroopMag_mV"] = df["batt_mV"] - df["boltDroop_mV"]
    df = decode_levels_and_faults(df)
    if (df["boltDroop_mV"] == 0).all():
        df["boltDroopMag_mV"] = 0
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
    total_time = float(df[time_col].iloc[-1] - df[time_col].iloc[0])
    offset = total_time * 0.005 if total_time > 0 else 0.001
    dot_size = 20
    for i, (col, param, unit) in enumerate(plots):
        ax = axes[i]
        ax.plot(df[time_col], df[col], label=param)
        ax.scatter(df[time_col][df["fault_brownout"]] - offset, df[col][df["fault_brownout"]], zorder=5, s=dot_size, color="red", label="Brownout")
        ax.scatter(df[time_col][df["fault_sound"]] + offset, df[col][df["fault_sound"]], zorder=5, s=dot_size, color="purple", label="Sound Op Fail")
        ax.scatter(df[time_col][df["fault_sound_brownout"]] + 2*offset, df[col][df["fault_sound_brownout"]], zorder=5, s=dot_size, color="blue", label="Sound Brownout")
        ax.scatter(df[time_col][df["fault_bolt"]] - offset, df[col][df["fault_bolt"]], zorder=5, s=dot_size, color="orange", label="Bolt Op Fail")
        ax.scatter(df[time_col][df["fault_bolt_brownout"]] - 2*offset, df[col][df["fault_bolt_brownout"]], zorder=5, s=dot_size, color="green", label="Bolt Brownout")
        ax.set_title(f"{param} vs Time Elapsed")
        ax.set_ylabel(unit)
        ax.grid(True)
        ax.legend()
        if col == "temp":
            ax.set_ylim(-30, 70)
        elif col == "batt_mV":
            ax.set_ylim(2000, 3400)
        elif col in ("soundDroopMag_mV", "boltDroopMag_mV"):
            ax.set_ylim(0, 2500)
        else:
            ax.set_ylim(0, 3400)
    for ax in axes:
        ax.set_xlabel("Time Elapsed (hours)")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close(fig)
    print(f"[ok] Wrote {png_path}")

def find_single_csv_png(dir_path):
    entries = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    csvs = [os.path.join(dir_path, f) for f in entries if f.lower().endswith(".csv")]
    pngs = [os.path.join(dir_path, f) for f in entries if f.lower().endswith(".png")]
    if len(csvs) == 1 and len(pngs) == 1:
        return csvs[0], pngs[0]
    return None, None

def main():
    if len(sys.argv) != 2:
        print("Usage: python regenerate_plots.py <base_directory>")
        sys.exit(1)
    base = sys.argv[1]
    if not os.path.isdir(base):
        print(f"Not a directory: {base}")
        sys.exit(1)
    subdirs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    if not subdirs:
        print("No subfolders found.")
        sys.exit(0)
    for d in sorted(subdirs):
        csv_path, png_path = find_single_csv_png(d)
        if csv_path and png_path:
            try:
                regenerate_png_from_csv(csv_path, png_path)
            except Exception as e:
                print(f"[err] {d}: {e}")
        else:
            print(f"[skip] {d}: needs exactly one CSV and one PNG")

if __name__ == "__main__":
    main()
