import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_monitor_results(log_dir="./models_robust/"):
    print(f"📊 Scanning for training logs in '{log_dir}'...")
    monitor_file = os.path.join(log_dir, "monitor.csv")
    
    if not os.path.exists(monitor_file):
        print(f"❌ Could not find {monitor_file}.")
        print("Make sure the training script has finished at least one full episode!")
        return

    # Stable Baselines3 monitor.csv files have a JSON dictionary on the very first line.
    # We use skiprows=1 to ignore that JSON line and read the actual CSV headers (r, l, t).
    try:
        df = pd.read_csv(monitor_file, skiprows=1)
    except Exception as e:
        print(f"❌ Error reading the CSV: {e}")
        return

    if df.empty:
        print("⚠️ The monitor.csv file is empty. No episodes have finished yet.")
        return

    # Extract metrics
    # 'r' = Episode Reward
    # 'l' = Episode Length (Timesteps)
    rewards = df['r']
    lengths = df['l']
    
    # Calculate cumulative timesteps for an accurate X-axis
    cumulative_timesteps = lengths.cumsum()

    # Calculate Rolling Average (Smoothing)
    # RL data is notoriously noisy. A dynamic window smooths out the spikes.
    window_size = max(5, len(df) // 10) 
    smoothed_rewards = rewards.rolling(window=window_size, min_periods=1).mean()

    print(f"📈 Generating Reward Curve for {len(df)} episodes ({cumulative_timesteps.iloc[-1]} timesteps)...")

    # Create the Plot
    plt.figure(figsize=(12, 7))
    
    # 1. Plot the raw, noisy data lightly in the background
    plt.plot(cumulative_timesteps, rewards, alpha=0.25, color='gray', label='Raw Episode Reward')
    
    # 2. Plot the smoothed trend line heavily in the foreground
    plt.plot(cumulative_timesteps, smoothed_rewards, color='darkblue', linewidth=2.5, 
             label=f'Smoothed Reward (Moving Avg: {window_size} eps)')

    # Formatting
    plt.title("PPO Agent Training Convergence: Emergency Vehicle Priority", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Total Training Timesteps", fontsize=12, fontweight='bold')
    plt.ylabel("Cumulative Episode Reward", fontsize=12, fontweight='bold')
    
    # Set X-axis limit to max timesteps for clean edges
    plt.xlim(0, cumulative_timesteps.iloc[-1])
    
    # Grid and Legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()

    # Save and Show
    filename = "ppo_learning_curve_robust.png"
    plt.savefig(filename, dpi=300)
    print(f"✅ Plot saved successfully as '{filename}'")
    
    plt.show()

if __name__ == "__main__":
    plot_monitor_results()