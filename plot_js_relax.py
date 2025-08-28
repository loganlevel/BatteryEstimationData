import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
CSV_FILE = "csvs/joulescope/20250827_204028-JS110-001152.csv"   # replace with your filename
TIME_COL = "#time"
VOLT_COL = "voltage"

def main():
    # Read CSV
    df = pd.read_csv(CSV_FILE)

    # Extract values
    time = df[TIME_COL]
    voltage = df[VOLT_COL]
    first_voltage = voltage.iloc[0]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(time, voltage, label="Voltage", linewidth=2)
    plt.axhline(y=first_voltage, color="r", linestyle="--", label=f"First voltage = {first_voltage:.4f} V")

    # Labels and legend
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Time vs Voltage")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

if __name__ == "__main__":
    main()
