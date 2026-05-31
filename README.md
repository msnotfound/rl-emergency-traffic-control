# 🚑 Emergency Vehicle Priority System

**Intelligent Traffic Signal Control using Reinforcement Learning**

## 📋 Overview

This project implements an AI-powered traffic signal control system designed to prioritize emergency vehicles (ambulances) while maintaining efficient overall traffic flow. Using Proximal Policy Optimization (PPO) and SUMO (Simulation of Urban MObility), the system learns optimal signal timing policies that significantly reduce emergency vehicle response times.

## 🎯 Problem Statement

Traditional fixed-time traffic signals cannot adapt to dynamic traffic conditions or prioritize emergency vehicles. This results in:
- Delayed emergency response times
- Increased risk to patients requiring urgent care
- Inefficient traffic flow during emergency situations

Our RL-based solution dynamically adjusts traffic signals to clear paths for ambulances while minimizing disruption to civilian traffic.

## 📸 Demo

<!-- TODO: add a screenshot or short GIF of the SUMO GUI: the trained agent giving the ambulance a green wave vs the fixed-time baseline. -->
_Demo coming soon — SUMO GUI showing the agent clearing an ambulance path through the intersection._

## 🛠️ Technical Stack

- **RL Framework**: Stable-Baselines3 (PPO)
- **Traffic Simulator**: SUMO 1.25.0
- **Environment Wrapper**: sumo-rl
- **Python**: 3.12
- **Key Libraries**: Gymnasium, PyTorch, NumPy

## 📁 Project Structure

```
rl-traffic-control/
├── train_optimized.py          # Optimized training script with VecNormalize
├── test_optimized.py            # Evaluation script with GUI visualization
├── train2.py                    # Basic training script
├── test2.py                     # Basic testing script
├── run_baseline_pure_traci.py  # Baseline fixed-time signals comparison
├── draft02.net.xml              # SUMO road network
├── draft02.rou.xml              # Civilian vehicle routes
├── ambulance.rou.xml            # Emergency vehicle configuration
├── vtypes.rou.xml               # Vehicle type definitions
└── draft02.sumocfg              # SUMO configuration file
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set SUMO_HOME environment variable
export SUMO_HOME="/usr/share/sumo"  # Adjust path as needed
```

### Running the Project

**1. Train the Optimized RL Agent**
```bash
python train_optimized.py
```
- Training runs for 100,000 timesteps
- Wall-clock: CPU-bound (no GPU needed); SUMO stepping dominates. Expect roughly tens of minutes to ~1–2 hours for the full run. _(TODO: drop in your measured time.)_
- Checkpoints saved every 10,000 steps to `./models/`
- Final model: `optimized_traffic_agent.zip`
- Normalization stats: `vec_normalize.pkl`

**2. Run Baseline Comparison**
```bash
python run_baseline_pure_traci.py
```
- Simulates fixed-time traffic signals
- Records ambulance travel time for comparison
- Visualizes simulation in SUMO GUI

**3. Test the Trained Agent**
```bash
python test_optimized.py
```
- Loads trained model and normalization stats
- Visualizes agent performance with SUMO GUI
- Tracks and reports ambulance travel time

## 🧠 Key Features

### Custom Reward Function
```python
reward = -1 * ((civilian_penalty × 0.1) + (ambulance_penalty × 5000))
```
- **Civilian Penalty**: Sum of accumulated waiting time across all lanes
- **Ambulance Penalty**: Massive penalty (5000) when ambulance speed < 1 m/s
- **Weighted Design**: Ambulance priority 50× more important than civilian flow

### Optimization Techniques

1. **VecNormalize**: Normalizes observations and rewards to prevent gradient explosion
2. **Big Brain Architecture**: 256×256 neural network (vs default 64×64)
3. **Fine-tuned Hyperparameters**:
   - Learning rate: 3e-4
   - Gamma: 0.995 (long-term planning)
   - Entropy coefficient: 0.01 (exploration)
   - n_steps: 2048 (experience collection)
   - GAE Lambda: 0.95 (variance smoothing)

### Environment Configuration
- **Simulation Time**: 1000 seconds per episode
- **Ambulance Spawn**: 120 seconds into simulation
- **Signal Constraints**: 5-60 seconds green time, 4 seconds yellow
- **Control Mode**: Single-agent (centralized control)

## 📊 Performance Metrics

The system tracks:
- **Ambulance Travel Time**: Primary metric (seconds from spawn to destination)
- **System Mean Waiting Time**: Average civilian vehicle delay
- **Episode Reward Mean**: Normalized reward (should converge to stable value)
- **Explained Variance**: How well the value function predicts returns

### Achieved Results
- **Baseline (Fixed-time)**: ~17 seconds ambulance travel time
- **RL Agent (Optimized)**: ~5 seconds ambulance travel time
- **Improvement**: 70% reduction in emergency response time
- **Impact**: The RL agent successfully prioritizes ambulance passage with minimal civilian traffic disruption

## 🔧 Configuration

### Training Parameters

Edit `train_optimized.py`:
```python
total_timesteps=100000      # Training duration
num_seconds=1000            # Episode length
learning_rate=3e-4          # Learning rate
checkpoint_freq=10000       # Save frequency
```

### Reward Tuning

Adjust weights in `custom_ambulance_reward()`:
```python
civilian_penalty * 0.1      # Civilian weight (default: 0.1)
ambulance_penalty = 5000    # Ambulance penalty (default: 5000)
```

## 📈 Monitoring Training

During training, watch for:
- `ep_rew_mean`: Should stabilize (not remain at -228k)
- `explained_variance`: Should increase from ~0 to 0.3-0.5
- `entropy_loss`: Should gradually decrease (exploration → exploitation)
- `policy_gradient_loss`: Should remain stable, not explode

## 🎮 Simulation Controls

When running with GUI (`use_gui=True`):
- **Space**: Pause/Resume
- **Mouse Wheel**: Zoom in/out
- **Right-click + Drag**: Pan view
- **Vehicle Click**: View individual vehicle details

## 📝 Requirements

```
sumo-rl
stable-baselines3
gymnasium
matplotlib
seaborn
shimmy
torch
```

## Status

Research prototype, paused. Single-intersection scenario validated in simulation: ~70% reduction in ambulance travel time (~17s → ~5s) versus a fixed-time baseline. Simulation-only — not validated for real-world deployment (see the note at the bottom).

## 🤝 Contributing

This is an academic/research project. Key areas for improvement:
- Multi-agent scenarios (multiple intersections)
- Real-world traffic pattern integration
- Transfer learning from simulation to reality
- Additional emergency vehicle types

## 📄 License

Released under the [MIT License](LICENSE).

## 👨‍💻 Author

MAYANK SAHU

## 🙏 Acknowledgments

- SUMO Development Team
- Stable-Baselines3 Contributors
- sumo-rl Library Maintainers

---

**Note**: This system is designed for simulation and research purposes. Real-world deployment would require extensive validation, safety testing, and regulatory approval.
