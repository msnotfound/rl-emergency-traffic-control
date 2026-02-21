import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def plot_traffic_metrics(prefix="robust_train"):
    print(f"📊 Scanning for episode CSV files starting with '{prefix}'...")
    
    # 1. Locate and sort files chronologically
    csv_files = [f for f in os.listdir(".") if f.startswith(prefix) and f.endswith(".csv")]
    
    if not csv_files:
        print(f"❌ No files found starting with '{prefix}'.")
        print("Make sure your training script has completed at least one episode and generated the CSVs.")
        return

    # Extract episode number
    def extract_ep_num(filename):
        match = re.search(r'_ep(\d+)\.csv', filename)
        return int(match.group(1)) if match else -1
    
    csv_files.sort(key=extract_ep_num)
    print(f"✅ Found {len(csv_files)} episode files. Processing data...")

    # 2. Extract and aggregate data
    episodes_data = []
    cumulative_steps = 0
    
    for file in csv_files:
        try:
            ep_num = extract_ep_num(file)
            if ep_num == -1: continue
                
            df = pd.read_csv(file)
            df = df[df["step"] > 0] # Filter initial state
            
            if len(df) > 0:
                steps_in_ep = len(df)
                cumulative_steps += steps_in_ep
                
                # Check if J4/J6 metrics exist (dynamic fallback depending on single/multi-agent)
                j4_wait = df["J4_accumulated_waiting_time"].mean() if "J4_accumulated_waiting_time" in df.columns else 0
                j6_wait = df["J6_accumulated_waiting_time"].mean() if "J6_accumulated_waiting_time" in df.columns else 0
                
                # Calculate episode metrics (mean performance across the entire episode)
                episodes_data.append({
                    "episode": ep_num,
                    "cumulative_steps": cumulative_steps,
                    "mean_waiting_time": df["system_mean_waiting_time"].mean(),
                    "mean_speed": df["system_mean_speed"].mean(),
                    "total_stopped": df["system_total_stopped"].mean(),
                    "j4_wait": j4_wait,
                    "j6_wait": j6_wait
                })
        except Exception as e:
            print(f"⚠️ Could not read {file}: {e}")

    master_df = pd.DataFrame(episodes_data)
    
    if master_df.empty:
        print("❌ No valid data extracted.")
        return

    # 3. Apply Rolling Averages for smoothing
    window_size = max(1, len(master_df) // 10)
    
    metrics_to_smooth = ["mean_waiting_time", "mean_speed", "total_stopped", "j4_wait", "j6_wait"]
    for metric in metrics_to_smooth:
        master_df[f"smoothed_{metric}"] = master_df[metric].rolling(window=window_size, min_periods=1).mean()

    print(f"📈 Generating Traffic Dashboard for {cumulative_steps} timesteps...")

    # 4. Create the Dashboard
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Traffic Flow Metrics: PPO Training Convergence", fontsize=18, fontweight="bold", y=0.98)

    # --- Plot 1: Mean Waiting Time ---
    axs[0, 0].plot(master_df["cumulative_steps"], master_df["mean_waiting_time"], alpha=0.2, color="red")
    axs[0, 0].plot(master_df["cumulative_steps"], master_df["smoothed_mean_waiting_time"], color="darkred", linewidth=2)
    axs[0, 0].set_title("System Mean Waiting Time (Lower is Better)", fontweight="bold")
    axs[0, 0].set_ylabel("Seconds")
    axs[0, 0].grid(True, linestyle="--", alpha=0.6)

    # --- Plot 2: Mean Speed ---
    axs[0, 1].plot(master_df["cumulative_steps"], master_df["mean_speed"], alpha=0.2, color="green")
    axs[0, 1].plot(master_df["cumulative_steps"], master_df["smoothed_mean_speed"], color="darkgreen", linewidth=2)
    axs[0, 1].set_title("System Mean Speed (Higher is Better)", fontweight="bold")
    axs[0, 1].set_ylabel("Speed (m/s)")
    axs[0, 1].grid(True, linestyle="--", alpha=0.6)

    # --- Plot 3: Total Stopped Vehicles ---
    axs[1, 0].plot(master_df["cumulative_steps"], master_df["total_stopped"], alpha=0.2, color="orange")
    axs[1, 0].plot(master_df["cumulative_steps"], master_df["smoothed_total_stopped"], color="darkorange", linewidth=2)
    axs[1, 0].set_title("Average Stopped Vehicles (Lower is Better)", fontweight="bold")
    axs[1, 0].set_xlabel("Total Training Timesteps")
    axs[1, 0].set_ylabel("Count")
    axs[1, 0].grid(True, linestyle="--", alpha=0.6)

    # --- Plot 4: J4 vs J6 Wait Time (if columns exist) ---
    has_j_metrics = master_df["j4_wait"].sum() > 0 or master_df["j6_wait"].sum() > 0
    if has_j_metrics:
        axs[1, 1].plot(master_df["cumulative_steps"], master_df["smoothed_j4_wait"], color="blue", linewidth=2, label="J4 Wait Time")
        axs[1, 1].plot(master_df["cumulative_steps"], master_df["smoothed_j6_wait"], color="purple", linewidth=2, label="J6 Wait Time")
        axs[1, 1].set_title("J4 vs J6 Accumulated Wait Time", fontweight="bold")
        axs[1, 1].legend()
    else:
        axs[1, 1].text(0.5, 0.5, "Specific intersection metrics not found in CSVs", 
                       horizontalalignment='center', verticalalignment='center', fontsize=12, color="gray")
        axs[1, 1].set_title("Intersection Specific Data", fontweight="bold")

    axs[1, 1].set_xlabel("Total Training Timesteps")
    axs[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = "ppo_traffic_metrics_dashboard_robust.png"
    plt.savefig(filename, dpi=300)
    print(f"✅ Dashboard saved successfully as '{filename}'")
    
    plt.show()

if __name__ == "__main__":
    # Ensure this matches the out_csv_name passed to your SumoEnvironment
    plot_traffic_metrics(prefix="robust_train")