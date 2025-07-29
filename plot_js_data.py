import pandas as pd
import os
import matplotlib.pyplot as plt

# Load CSV data
csv_path = 'csvs/BE-1-JS.csv'
df = pd.read_csv(csv_path)
csv_filename = os.path.splitext(os.path.basename(csv_path))[0]

# Convert units
df['elapsed_time_hr'] = df['elapsed_time'] / 3600  # seconds to hours
df['charge_mAh'] = df['charge_C'] / 3.6     # C to mAh (1 C = 1 A·s; 1 mAh = 3.6 C)

# Create plots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Battery Logging Overview")

# Plot 1: Charge vs Elapsed Time
axs[0, 0].plot(df['elapsed_time_hr'], df['charge_mAh'], marker='o')
axs[0, 0].set_title("Charge (mAh) vs Time (hr)")
axs[0, 0].set_xlabel("Time (hours)")
axs[0, 0].set_ylabel("Charge (mAh)")
axs[0, 0].grid(True)

# Plot 2: Avg Voltage vs Elapsed Time
axs[0, 1].plot(df['elapsed_time_hr'], df['avg_voltage_V'], marker='o', color='green')
axs[0, 1].set_title("Avg Voltage (V) vs Time (hr)")
axs[0, 1].set_xlabel("Time (hours)")
axs[0, 1].set_ylabel("Avg Voltage (V)")
axs[0, 1].grid(True)

# Plot 3: Min Voltage vs Elapsed Time
axs[1, 0].plot(df['elapsed_time_hr'], df['min_voltage_V'], marker='o', color='red')
axs[1, 0].set_title("Min Voltage (V) vs Time (hr)")
axs[1, 0].set_xlabel("Time (hours)")
axs[1, 0].set_ylabel("Min Voltage (V)")
axs[1, 0].grid(True)

# Plot 4: Avg Current vs Elapsed Time
axs[1, 1].plot(df['elapsed_time_hr'], df['avg_current_A'], marker='o', color='purple')
axs[1, 1].set_title("Avg Current (A) vs Time (hr)")
axs[1, 1].set_xlabel("Time (hours)")
axs[1, 1].set_ylabel("Avg Current (A)")
axs[1, 1].grid(True)

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'plots/{csv_filename}_summary.png')
plt.close()
