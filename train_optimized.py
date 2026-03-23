import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import sumo_rl
import traci
import os
import torch.nn as nn
import zlib  # 🚨 THE FIX: Added zlib for deterministic hashing 🚨

class AmbulanceObservationWrapper(gym.ObservationWrapper):
    """
    EXTREMELY IMPORTANT FOR ROBUSTNESS:
    This wrapper modifies the 'State' the AI sees.
    It adds a 1 or 0 for every lane. 1 means the ambulance is in that lane.
    This allows the AI to 'see' the ambulance regardless of the seed.
    """
    def __init__(self, env):
        super().__init__(env)
        # We add 1 extra bit per lane to the observation space

        # Standard sumo-rl obs for 2 intersections is roughly 40-50 dims. 
        # We append the 'Ambulance Presence' per lane.
        orig_shape = self.observation_space.shape[0]
        # In draft02.net.xml, you have roughly 16 incoming lanes across J4 and J6
        # To keep it simple and robust, we add 20 extra slots for emergency tracking.
        self.add_dim = 20 
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(orig_shape + self.add_dim,), dtype=np.float32
        )

    def observation(self, obs):
        # Create a blank vector for ambulance presence
        ambulance_pos = np.zeros(self.add_dim, dtype=np.float32)
        try:
            veh_list = traci.vehicle.getIDList()
            if "hero_ambulance" in veh_list:
                lane_id = traci.vehicle.getLaneID("hero_ambulance")
                # 🚨 THE FIX: zlib.crc32 guarantees the lane name maps to the EXACT 
                # same index during both training and testing, every single time.
                lane_idx = zlib.crc32(lane_id.encode('utf-8')) % self.add_dim
                ambulance_pos[lane_idx] = 1.0
        except:
            pass
        # Combine the standard traffic data with the explicit ambulance location
        return np.concatenate([obs, ambulance_pos])

def custom_ambulance_reward(traffic_signal):
    lane_waits = traffic_signal.get_accumulated_waiting_time_per_lane()
    civilian_penalty = sum(lane_waits)
    
    ambulance_penalty = 0
    try:
        if "hero_ambulance" in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed("hero_ambulance")
            if speed < 1.0:
                ambulance_penalty += 5000 
    except:
        pass

    return -1 * ((civilian_penalty * 0.7) + ambulance_penalty)

def train_robust():
    log_dir = "./models_robust/"
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. Create Environment WITHOUT a fixed seed
    env = sumo_rl.SumoEnvironment(
        net_file="draft02.net.xml",
        route_file="vtypes.rou.xml,traffic_dense.rou.xml,ambulance_s.rou.xml",
        out_csv_name="robust_train",
        use_gui=False,
        num_seconds=1000,
        yellow_time=4,
        min_green=5,
        max_green=60,
        single_agent=True,
        reward_fn=custom_ambulance_reward
        # NO SEED HERE -> It will be different every episode!
    )

    # 2. Apply the Robustness Wrapper
    env = AmbulanceObservationWrapper(env)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # 3. Increase learning complexity for a 'Universal Brain'
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[512, 512], vf=[512, 512]) # Bigger brain for more patterns
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2e-4, # Slightly slower learning for better stability
        n_steps=4096,       # See more traffic per update
        batch_size=128,
        ent_coef=0.02,      # 🚨 HIGHER EXPLORATION 🚨
        policy_kwargs=policy_kwargs
    )

    print("🚀 Starting ROBUST training (No seed restriction)...")
    # Training for 200k steps because 'Universal' learning takes longer than 'Memorization'
    model.learn(total_timesteps=200000)

    model.save("robust_traffic_agent")
    env.save("robust_vec_normalise.pkl")
    print("✅ Robust Model Saved.")

if __name__ == "__main__":
    train_robust()