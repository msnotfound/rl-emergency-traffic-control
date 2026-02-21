import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import sumo_rl
import traci
import os
import torch.nn as nn

def custom_ambulance_reward_s(traffic_signal):
    """
    Enhanced Weighted Reward for S-Scenario:
    - Penalize Ambulance delay 10x more than normal cars
    - Track performance across both intersections
    """
    # 1. Civilian Traffic Penalty
    lane_waits = traffic_signal.get_accumulated_waiting_time_per_lane()
    civilian_penalty = sum(lane_waits)
    
    # 2. Emergency Penalty
    ambulance_penalty = 0
    try:
        vehicle_list = traci.vehicle.getIDList()
        for veh_id in vehicle_list:
            if traci.vehicle.getTypeID(veh_id) == "ambulance_type":
                speed = traci.vehicle.getSpeed(veh_id)
                if speed < 1.0:
                    # MASSIVE penalty to force immediate reaction
                    ambulance_penalty += 5000 
    except:
        pass

    # Combine: Balance ambulance priority with civilian traffic flow
    reward = -1 * ((civilian_penalty * 0.7) + ambulance_penalty)
    return reward

def train_scenario_s():
    net_file = "draft02.net.xml"
    route_file = "vtypes.rou.xml,traffic_dense.rou.xml,ambulance_s.rou.xml"
    
    # Create models directory for S-scenario
    os.makedirs("models_s", exist_ok=True)
    
    # Define the Checkpoint: Save every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models_s/",
        name_prefix="rl_model_scenario_s"
    )
    
    # 1. Create the Environment (controls all traffic signals: J4 and J6)
    env = sumo_rl.SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        fixed_ts=False,
        out_csv_name="traffic_result_s",
        use_gui=False,
        num_seconds=1000,  # Longer episodes for complex scenario
        yellow_time=4,
        min_green=5,
        max_green=60,
        single_agent=True,  # Single agent controlling all traffic signals
        reward_fn=custom_ambulance_reward_s
    )

    # 2. VECTORIZE & NORMALIZE
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    print("🧠 Initializing S-Scenario PPO Agent (Controlling J4 & J6)...")
    
    # 3. Define a Custom "Big Brain" Policy for complex scenario
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])  # Two layers of 256 neurons
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        # --- HYPERPARAMETER TUNING ---
        learning_rate=3e-4,      # 0.0003 (Standard Stable Value)
        gamma=0.995,             # Care about long-term future
        gae_lambda=0.95,         # Smooth variance
        clip_range=0.2,          # Don't make wild changes
        ent_coef=0.01,           # Explore more!
        n_steps=2048,            # Collect more experience before updating
        batch_size=64,           # Smaller batches for better gradient updates
        policy_kwargs=policy_kwargs
    )

    print("🚀 Starting S-Scenario Training (300k Steps)...")
    print("   - Dual-intersection control (J4 and J6)")
    print("   - S-shaped ambulance route")
    print("   - Heavy traffic (300+ vehicles)")
    model.learn(total_timesteps=300000, callback=checkpoint_callback)

    # SAVE BOTH MODEL AND NORMALIZATION STATS
    model.save("models_s/optimized_traffic_agent_s")
    env.save("models_s/vec_normalize_s.pkl")
    print("✅ S-Scenario Model & Normalization Stats saved to models_s/")

if __name__ == "__main__":
    train_scenario_s()
