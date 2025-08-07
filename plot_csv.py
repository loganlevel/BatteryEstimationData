import sys
import pandas as pd
import matplotlib.pyplot as plt
import os


def main():
    CSV_FILE = "csvs/Maven_B05_BE-7_20250801_droop2_load2_loaddur1800_relaxdur1800_boltcheckstrue.csv"
    df = pd.read_csv(CSV_FILE)
    filename = os.path.splitext(os.path.basename(CSV_FILE))[0]

    # Prepare plotting
    os.makedirs("plots", exist_ok=True)
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

        ax.scatter(df[time_col][df["fault_bolt"]],
                   df[col][df["fault_bolt"]],
                   color="orange", label="Bolt Fail", zorder=5, s=dot_size)

        ax.scatter(df[time_col][df["fault_sound"]] + offset,
                   df[col][df["fault_sound"]],
                   color="purple", label="Sound Fail", zorder=5, s=dot_size)

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
    output_png = f"plots/{filename}_all_plots.png"
    plt.savefig(output_png)
    print(f"Saved all plots as: {output_png}")

if __name__ == "__main__":
    main()
