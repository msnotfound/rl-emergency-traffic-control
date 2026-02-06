import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparison():
    # 1. SETUP DATA
    # ---------------------------------------------------------
    # REPLACE THIS NUMBER with the result from 'run_baseline.py'
    baseline_time = 75.0  
    
    # Load the RL results
    df = pd.read_csv("traffic_result.csv")
    
    # Calculate RL Average (Last 10 episodes to show final performance)
    # Assuming 'system_mean_waiting_time' is the metric column
    rl_time = df['system_mean_waiting_time'].tail(10).mean()
    # ---------------------------------------------------------

    # 2. CREATE THE LEARNING CURVE (Line Graph)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df, x='step', y='system_mean_waiting_time', label='RL Agent')
    plt.axhline(y=baseline_time, color='r', linestyle='--', label='Fixed-Time Baseline')
    plt.title("Learning Curve: Traffic Waiting Time")
    plt.xlabel("Training Steps")
    plt.ylabel("Avg Waiting Time (s)")
    plt.legend()
    plt.grid(True)

    # 3. CREATE THE COMPARISON (Bar Chart)
    plt.subplot(1, 2, 2)
    data = {'System': ['Fixed-Time (Baseline)', 'PPO Agent (Ours)'], 
            'Time (s)': [baseline_time, rl_time]}
    
    sns.barplot(x='System', y='Time (s)', data=data, palette=['red', 'green'])
    plt.title("Final Performance Comparison")
    plt.ylabel("Avg Waiting Time (Lower is Better)")
    
    # Add text labels on bars
    for i, v in enumerate([baseline_time, rl_time]):
        plt.text(i, v + 1, str(round(v, 2)), ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_comparison()