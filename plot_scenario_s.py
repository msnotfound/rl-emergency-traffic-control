import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import numpy as np
from stable_baselines3.common.results_plotter import plot_results
import matplotlib.pyplot as plt

# #  Plot the average reward over time
# # plot_results([log_dir], 10000, results_plotter.X_TIMESTEPS, "SB3 CartPole Learning Curve")
# # plt.show()


# def plot_scenario_s():
#     """
#     Enhanced plotting for S-Scenario with:
#     - Learning curve
#     - Ambulance comparison (baseline vs RL)
#     - Civilian traffic comparison
#     - Per-intersection analysis
#     - Route segment timing
#     """

#     print("📊 Loading S-Scenario Results...")

#     # 1. LOAD BASELINE RESULTS
#     # ---------------------------------------------------------
#     if os.path.exists("baseline_result_s.txt"):
#         with open("baseline_result_s.txt", "r") as f:
#             baseline = json.load(f)
#         print(f"✅ Loaded baseline - Ambulance: {baseline['ambulance_total_time']}s")
#     else:
#         print("⚠️ baseline_result_s.txt not found! Run 'baseline_scenario_s.py' first.")
#         baseline = {
#             "ambulance_total_time": 0,
#             "civilian_avg_wait": 0,
#             "j4_crossing_time": 0,
#             "j6_crossing_time": 0,
#             "total_vehicles": 0,
#         }

#     # 2. LOAD OPTIMIZED RESULTS
#     # ---------------------------------------------------------
#     if os.path.exists("optimized_result_s.txt"):
#         with open("optimized_result_s.txt", "r") as f:
#             optimized = json.load(f)
#         print(f"✅ Loaded optimized - Ambulance: {optimized['ambulance_total_time']}s")
#     else:
#         print("⚠️ optimized_result_s.txt not found! Run 'test_scenario_s.py' first.")
#         optimized = {
#             "ambulance_total_time": 0,
#             "civilian_avg_wait": 0,
#             "j4_crossing_time": 0,
#             "j6_crossing_time": 0,
#             "total_vehicles": 0,
#         }

#     # 3. LOAD TRAINING RESULTS
#     # ---------------------------------------------------------
#     # Combine all CSV files from training
#     csv_files = [
#         f
#         for f in os.listdir(".")
#         if f.startswith("traffic_result_s_conn0_ep") and f.endswith(".csv")
#     ]

#     if csv_files:
#         # Read and concatenate all CSV files
#         dfs = []
#         for file in csv_files:
#             try:
#                 df_temp = pd.read_csv(file)
#                 # Filter for steps (exclude step=0 which is initial state)
#                 df_temp = df_temp[df_temp["step"] > 0]
#                 dfs.append(df_temp)
#             except Exception as e:
#                 print(f"Warning: Could not read {file}: {e}")

#         if dfs:
#             df = pd.concat(dfs, ignore_index=True)
#             # Calculate episode-level metrics
#             df["episode"] = df.groupby("step").cumcount() + 1
#             df["episode_mean_waiting"] = df.groupby("episode")[
#                 "system_mean_waiting_time"
#             ].transform("mean")
#             has_training_data = True
#             print(
#                 f"✅ Loaded training data - {len(df)} steps across {df['episode'].nunique()} episodes"
#             )
#         else:
#             print("⚠️ No valid training data found!")
#             has_training_data = False
#     else:
#         print("⚠️ No CSV files found! Skipping learning curve.")
#         has_training_data = False

#     # ---------------------------------------------------------
#     # 4. CREATE THE PLOTS
#     # ---------------------------------------------------------

#     # Determine layout based on available data
#     if has_training_data and baseline["ambulance_total_time"] > 0:
#         # Full layout: 2 rows, 3 columns
#         fig = plt.figure(figsize=(20, 12))

#         # Row 1: Learning curve, ambulance comparison, civilian comparison
#         ax1 = plt.subplot(2, 3, 1)
#         ax2 = plt.subplot(2, 3, 2)
#         ax3 = plt.subplot(2, 3, 3)

#         # Row 2: Per-intersection analysis, route segment timing, congestion
#         ax4 = plt.subplot(2, 3, 4)
#         ax5 = plt.subplot(2, 3, 5)
#         ax6 = plt.subplot(2, 3, 6)

#         # Plot 1: Learning Curve
#         plt.sca(ax1)
#         sns.lineplot(
#             data=df,
#             x="step",
#             y="system_mean_waiting_time",
#             label="RL Agent",
#             color="green",
#         )
#         plt.axhline(
#             y=baseline["civilian_avg_wait"],
#             color="r",
#             linestyle="--",
#             label="Fixed-Time Baseline",
#         )
#         plt.title(
#             "Learning Curve: Traffic Waiting Time", fontsize=14, fontweight="bold"
#         )
#         plt.xlabel("Training Steps")
#         plt.ylabel("Avg Waiting Time (s)")
#         plt.legend()
#         plt.grid(True, alpha=0.3)

#     else:
#         # Simplified layout: 2 rows, 2 columns
#         fig = plt.figure(figsize=(16, 12))
#         ax2 = plt.subplot(2, 2, 1)
#         ax3 = plt.subplot(2, 2, 2)
#         ax4 = plt.subplot(2, 2, 3)
#         ax5 = plt.subplot(2, 2, 4)

#     # Plot 2: Ambulance Transit Time Comparison
#     plt.sca(ax2)
#     data = {
#         "System": ["Fixed-Time\n(Baseline)", "PPO Agent\n(Optimized)"],
#         "Time (s)": [
#             baseline["ambulance_total_time"],
#             optimized["ambulance_total_time"],
#         ],
#     }
#     bars = plt.bar(
#         data["System"],
#         data["Time (s)"],
#         color=["#ff6b6b", "#51cf66"],
#         edgecolor="black",
#         linewidth=1.5,
#     )
#     plt.title("Ambulance Transit Time (S-Route)", fontsize=14, fontweight="bold")
#     plt.ylabel("Transit Time (seconds)", fontsize=12)
#     plt.ylim(0, max(data["Time (s)"]) * 1.2 if max(data["Time (s)"]) > 0 else 100)

#     # Add value labels on bars
#     if baseline["ambulance_total_time"] > 0 and optimized["ambulance_total_time"] > 0:
#         for i, (bar, v) in enumerate(zip(bars, data["Time (s)"])):
#             plt.text(
#                 bar.get_x() + bar.get_width() / 2,
#                 v + 2,
#                 f"{v:.1f}s",
#                 ha="center",
#                 va="bottom",
#                 fontweight="bold",
#                 fontsize=11,
#             )

#         # Calculate improvement
#         improvement = (
#             (baseline["ambulance_total_time"] - optimized["ambulance_total_time"])
#             / baseline["ambulance_total_time"]
#         ) * 100
#         plt.text(
#             0.5,
#             max(data["Time (s)"]) * 1.05,
#             f"Improvement: {improvement:.1f}%",
#             ha="center",
#             fontsize=12,
#             fontweight="bold",
#             bbox=dict(
#                 boxstyle="round", facecolor="yellow", alpha=0.7, edgecolor="black"
#             ),
#         )

#     plt.grid(True, alpha=0.3, axis="y")

#     # Plot 3: Civilian Waiting Time Comparison
#     plt.sca(ax3)
#     civilian_data = {
#         "System": ["Fixed-Time\n(Baseline)", "PPO Agent\n(Optimized)"],
#         "Wait Time (s)": [
#             baseline["civilian_avg_wait"],
#             optimized["civilian_avg_wait"],
#         ],
#     }
#     bars = plt.bar(
#         civilian_data["System"],
#         civilian_data["Wait Time (s)"],
#         color=["#ff6b6b", "#51cf66"],
#         edgecolor="black",
#         linewidth=1.5,
#     )
#     plt.title("Civilian Avg Waiting Time", fontsize=14, fontweight="bold")
#     plt.ylabel("Avg Waiting Time (seconds)", fontsize=12)
#     plt.ylim(
#         0,
#         max(civilian_data["Wait Time (s)"]) * 1.2
#         if max(civilian_data["Wait Time (s)"]) > 0
#         else 50,
#     )

#     # Add value labels
#     if baseline["civilian_avg_wait"] > 0 and optimized["civilian_avg_wait"] > 0:
#         for i, (bar, v) in enumerate(zip(bars, civilian_data["Wait Time (s)"])):
#             plt.text(
#                 bar.get_x() + bar.get_width() / 2,
#                 v + 1,
#                 f"{v:.1f}s",
#                 ha="center",
#                 va="bottom",
#                 fontweight="bold",
#                 fontsize=11,
#             )

#         # Calculate improvement
#         civilian_improvement = (
#             (baseline["civilian_avg_wait"] - optimized["civilian_avg_wait"])
#             / baseline["civilian_avg_wait"]
#         ) * 100
#         plt.text(
#             0.5,
#             max(civilian_data["Wait Time (s)"]) * 1.05,
#             f"Change: {civilian_improvement:+.1f}%",
#             ha="center",
#             fontsize=12,
#             fontweight="bold",
#             bbox=dict(
#                 boxstyle="round",
#                 facecolor="lightgreen" if civilian_improvement > 0 else "lightcoral",
#                 alpha=0.7,
#                 edgecolor="black",
#             ),
#         )

#     plt.grid(True, alpha=0.3, axis="y")

#     # Plot 4: Per-Intersection Analysis (J4 vs J6)
#     plt.sca(ax4)
#     intersections = ["J4 Crossing", "J6 Crossing"]
#     baseline_times = [baseline["j4_crossing_time"], baseline["j6_crossing_time"]]
#     optimized_times = [optimized["j4_crossing_time"], optimized["j6_crossing_time"]]

#     x = np.arange(len(intersections))
#     width = 0.35

#     bars1 = plt.bar(
#         x - width / 2,
#         baseline_times,
#         width,
#         label="Baseline",
#         color="#ff6b6b",
#         edgecolor="black",
#         linewidth=1.5,
#     )
#     bars2 = plt.bar(
#         x + width / 2,
#         optimized_times,
#         width,
#         label="RL Agent",
#         color="#51cf66",
#         edgecolor="black",
#         linewidth=1.5,
#     )

#     plt.xlabel("Intersection", fontsize=12)
#     plt.ylabel("Crossing Time (seconds)", fontsize=12)
#     plt.title("Per-Intersection Performance", fontsize=14, fontweight="bold")
#     plt.xticks(x, intersections)
#     plt.legend()
#     plt.grid(True, alpha=0.3, axis="y")

#     # Add value labels
#     for bars in [bars1, bars2]:
#         for bar in bars:
#             height = bar.get_height()
#             if height > 0:
#                 plt.text(
#                     bar.get_x() + bar.get_width() / 2.0,
#                     height + 0.5,
#                     f"{height:.1f}s",
#                     ha="center",
#                     va="bottom",
#                     fontsize=9,
#                 )

#     # Plot 5: Route Segment Timing Breakdown
#     plt.sca(ax5)

#     # Create stacked bar chart showing route segments
#     segments = ["Start → J4", "J4 → J6"]
#     baseline_segments = [baseline["j4_crossing_time"], baseline["j6_crossing_time"]]
#     optimized_segments = [optimized["j4_crossing_time"], optimized["j6_crossing_time"]]

#     x_pos = [0, 1]

#     # Baseline stacked bar
#     plt.bar(
#         0,
#         baseline_segments[0],
#         color="#ffa07a",
#         edgecolor="black",
#         linewidth=1.5,
#         label="Segment 1 (→J4)",
#     )
#     plt.bar(
#         0,
#         baseline_segments[1],
#         bottom=baseline_segments[0],
#         color="#ff6b6b",
#         edgecolor="black",
#         linewidth=1.5,
#         label="Segment 2 (J4→J6)",
#     )

#     # Optimized stacked bar
#     plt.bar(1, optimized_segments[0], color="#90ee90", edgecolor="black", linewidth=1.5)
#     plt.bar(
#         1,
#         optimized_segments[1],
#         bottom=optimized_segments[0],
#         color="#51cf66",
#         edgecolor="black",
#         linewidth=1.5,
#     )

#     plt.ylabel("Time (seconds)", fontsize=12)
#     plt.title("S-Route Segment Timing", fontsize=14, fontweight="bold")
#     plt.xticks([0, 1], ["Baseline", "RL Agent"])
#     plt.legend(loc="upper right")
#     plt.grid(True, alpha=0.3, axis="y")

#     # Add total time labels
#     baseline_total = sum(baseline_segments)
#     optimized_total = sum(optimized_segments)
#     if baseline_total > 0:
#         plt.text(
#             0,
#             baseline_total + 2,
#             f"Total: {baseline_total:.1f}s",
#             ha="center",
#             fontweight="bold",
#             fontsize=10,
#         )
#     if optimized_total > 0:
#         plt.text(
#             1,
#             optimized_total + 2,
#             f"Total: {optimized_total:.1f}s",
#             ha="center",
#             fontweight="bold",
#             fontsize=10,
#         )

#     # Plot 6: Traffic Volume Comparison (if we have full layout)
#     if has_training_data and baseline["ambulance_total_time"] > 0:
#         plt.sca(ax6)

#         # Show vehicle counts
#         vehicle_data = {
#             "Metric": ["Total Vehicles\nProcessed", "Avg Waiting\nTime (s)"],
#             "Baseline": [baseline["total_vehicles"], baseline["civilian_avg_wait"]],
#             "RL Agent": [optimized["total_vehicles"], optimized["civilian_avg_wait"]],
#         }

#         # Create grouped bar chart
#         x = np.arange(len(vehicle_data["Metric"]))
#         width = 0.35

#         plt.bar(
#             x - width / 2,
#             vehicle_data["Baseline"],
#             width,
#             label="Baseline",
#             color="#ff6b6b",
#             edgecolor="black",
#             linewidth=1.5,
#         )
#         plt.bar(
#             x + width / 2,
#             vehicle_data["RL Agent"],
#             width,
#             label="RL Agent",
#             color="#51cf66",
#             edgecolor="black",
#             linewidth=1.5,
#         )

#         plt.ylabel("Value", fontsize=12)
#         plt.title("Traffic Volume & Efficiency", fontsize=14, fontweight="bold")
#         plt.xticks(x, vehicle_data["Metric"])
#         plt.legend()
#         plt.grid(True, alpha=0.3, axis="y")

#     # Overall title
#     fig.suptitle(
#         "S-Scenario Analysis: Dual-Intersection Ambulance Routing",
#         fontsize=16,
#         fontweight="bold",
#         y=0.995,
#     )

#     plt.tight_layout()

#     # Save the figure
#     output_path = "s_scenario_results.png"
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     print(f"✅ Plot saved to: {output_path}")

#     plt.show()

#     # Print summary
#     print("\n" + "=" * 60)
#     print("S-SCENARIO SUMMARY")
#     print("=" * 60)
#     if baseline["ambulance_total_time"] > 0 and optimized["ambulance_total_time"] > 0:
#         improvement = (
#             (baseline["ambulance_total_time"] - optimized["ambulance_total_time"])
#             / baseline["ambulance_total_time"]
#         ) * 100
#         print(f"Ambulance Time Improvement: {improvement:+.1f}%")
#         print(f"  Baseline: {baseline['ambulance_total_time']:.1f}s")
#         print(f"  RL Agent: {optimized['ambulance_total_time']:.1f}s")
#         print()

#         civilian_change = (
#             (baseline["civilian_avg_wait"] - optimized["civilian_avg_wait"])
#             / baseline["civilian_avg_wait"]
#         ) * 100
#         print(f"Civilian Wait Time Change: {civilian_change:+.1f}%")
#         print(f"  Baseline: {baseline['civilian_avg_wait']:.1f}s")
#         print(f"  RL Agent: {optimized['civilian_avg_wait']:.1f}s")
#         print()

#         print("Per-Intersection Performance:")
#         print(
#             f"  J4 - Baseline: {baseline['j4_crossing_time']:.1f}s, RL: {optimized['j4_crossing_time']:.1f}s"
#         )
#         print(
#             f"  J6 - Baseline: {baseline['j6_crossing_time']:.1f}s, RL: {optimized['j6_crossing_time']:.1f}s"
#         )
#     print("=" * 60)


# if __name__ == "__main__":
#     plot_scenario_s()
# 3. LOAD TRAINING RESULTS (FIXED FOR MULTIPLE CSVs)
    # ---------------------------------------------------------
import re # Make sure to add 'import re' at the top of your script

    # Find all training CSVs
csv_files = [
    f for f in os.listdir(".")
    if f.startswith("traffic_result_s") and f.endswith(".csv")
    ]

if csv_files:
        episode_data = []
        cumulative_steps = 0
        
        # Helper function to extract episode number from filename for correct sorting
        def get_ep_num(filename):
            match = re.search(r'_ep(\d+)', filename)
            return int(match.group(1)) if match else 0
            
        # Sort files chronologically (ep1, ep2, ep3...)
        csv_files.sort(key=get_ep_num)

        for file in csv_files:
            try:
                df_temp = pd.read_csv(file)
                # Filter out the initial 0 step
                df_temp = df_temp[df_temp["step"] > 0] 
                
                if len(df_temp) > 0:
                    ep_num = get_ep_num(file)
                    # Get the average waiting time for this specific episode
                    avg_wait = df_temp['system_mean_waiting_time'].mean()
                    
                    # Track total steps across all episodes for the X-axis
                    steps_in_ep = len(df_temp)
                    cumulative_steps += steps_in_ep
                    
                    episode_data.append({
                        'episode': ep_num,
                        'cumulative_step': cumulative_steps,
                        'system_mean_waiting_time': avg_wait
                    })
            except Exception as e:
                print(f"Warning: Could not read {file}: {e}")

        if episode_data:
            # Create a clean dataframe with one row per episode
            df = pd.DataFrame(episode_data)
            has_training_data = True
            print(f"✅ Loaded training data - {cumulative_steps} total steps across {len(df)} episodes")
        else:
            print("⚠️ No valid training data found inside the CSVs!")
            has_training_data = False
else:
        print("⚠️ No CSV files found! Skipping learning curve.")
        has_training_data = False

    # ---------------------------------------------------------
    # 4. CREATE THE PLOTS
    # ---------------------------------------------------------

    # Determine layout based on available data
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_sumo_learning_curve():
    print("📊 Scanning for episode CSV files...")
    
    # 1. Find all CSV files generated by the training script
    prefix = "traffic_result_s"
    csv_files = [f for f in os.listdir(".") if f.startswith(prefix) and f.endswith(".csv")]
    
    if not csv_files:
        print(f"❌ No files found starting with '{prefix}'. Are you in the right directory?")
        return

    # 2. Extract episode number and sort chronologically
    # This prevents 'ep10' from loading before 'ep2'
    def extract_ep_num(filename):
        match = re.search(r'_ep(\d+)\.csv', filename)
        return int(match.group(1)) if match else -1
    
    csv_files.sort(key=extract_ep_num)
    
    print(f"✅ Found {len(csv_files)} episode files. Processing data...")

    # 3. Aggregate data per episode
    episodes_data = []
    cumulative_steps = 0
    
    for file in csv_files:
        try:
            ep_num = extract_ep_num(file)
            if ep_num == -1: continue
                
            df = pd.read_csv(file)
            # Filter out the initial '0' step
            df = df[df["step"] > 0]
            
            if len(df) > 0:
                # Get the average waiting time for this episode
                # You can also change this to 'system_total_stopped' if you prefer
                avg_wait = df["system_mean_waiting_time"].mean()
                
                steps_in_ep = len(df)
                cumulative_steps += steps_in_ep
                
                episodes_data.append({
                    "episode": ep_num,
                    "cumulative_steps": cumulative_steps,
                    "avg_waiting_time": avg_wait
                })
        except Exception as e:
            print(f"⚠️ Could not read {file}: {e}")

    # 4. Create a master DataFrame
    master_df = pd.DataFrame(episodes_data)
    
    if master_df.empty:
        print("❌ No valid data extracted from the CSVs.")
        return

    # 5. Apply a Rolling Average (Crucial for RL graphs)
    # RL data is extremely spiky. A rolling average smooths it out into a readable trend.
    window_size = min(50, len(master_df) // 10) # Adjust window based on total episodes
    window_size = max(1, window_size) # Prevent window size of 0
    master_df["smoothed_wait"] = master_df["avg_waiting_time"].rolling(window=window_size, min_periods=1).mean()

    print(f"📈 Generating plot for {cumulative_steps} total timesteps...")

    # 6. Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot the raw spiky data lightly in the background
    plt.plot(master_df["cumulative_steps"], master_df["avg_waiting_time"], 
             alpha=0.25, color="#ff9999", label="Raw Episode Mean")
    
    # Plot the smoothed solid line in the foreground
    plt.plot(master_df["cumulative_steps"], master_df["smoothed_wait"], 
             color="#cc0000", linewidth=2.5, label=f"Trend ({window_size}-Episode Average)")

    # Formatting
    plt.title("PPO Agent Learning Curve: Dual-Intersection S-Route", fontsize=16, fontweight="bold")
    plt.xlabel("Total Training Timesteps", fontsize=12)
    plt.ylabel("System Mean Waiting Time (s)", fontsize=12)
    
    # Optional: Highlight the fact that it goes down
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper right", fontsize=12)
    
    plt.tight_layout()

    # Save and show
    plt.savefig("ppo_learning_curve_s_scenario.png", dpi=300)
    print("✅ Plot saved successfully as 'ppo_learning_curve_s_scenario.png'")
    
    plt.show()

if __name__ == "__main__":
    plot_sumo_learning_curve()