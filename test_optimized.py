import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sumo_rl
import traci
import os
import time

class AmbulanceObservationWrapper(gym.ObservationWrapper):
    """
    🚨 REQUIRED FOR ROBUST TESTING 🚨
    This must exactly match the wrapper used during training so the AI 
    receives the exact same number of inputs (Traffic Data + 20 Lane Trackers).
    """
    def __init__(self, env):
        super().__init__(env)
        orig_shape = self.observation_space.shape[0]
        self.add_dim = 20 
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(orig_shape + self.add_dim,), dtype=np.float32
        )

    def observation(self, obs):
        ambulance_pos = np.zeros(self.add_dim, dtype=np.float32)
        try:
            veh_list = traci.vehicle.getIDList()
            if "hero_ambulance" in veh_list:
                lane_id = traci.vehicle.getLaneID("hero_ambulance")
                lane_idx = abs(hash(lane_id)) % self.add_dim
                ambulance_pos[lane_idx] = 1.0
        except:
            pass
        return np.concatenate([obs, ambulance_pos])


def test_optimized():
    print("🚀 Loading Robust Universal Trained Model...")
    
    # 1. Setup Same Environment (NO SEED LOCKED THIS TIME! 🎲)
    env = sumo_rl.SumoEnvironment(
        net_file="draft02.net.xml",
        route_file="vtypes.rou.xml,traffic_dense.rou.xml,ambulance_s.rou.xml",
        out_csv_name=None,
        use_gui=True,
        num_seconds=1000,
        fixed_ts=False,
        yellow_time=4,
        min_green=5,
        max_green=60,
        single_agent=True
        # NO SEED: Let the traffic be completely random!
    )
    
    # 2. Apply the Robust Wrapper First!
    env = AmbulanceObservationWrapper(env)
    
    # 3. Re-Apply Normalization Wrapper
    env = DummyVecEnv([lambda: env])
    
    # Check if normalization file exists
    if not os.path.exists("robust_vec_normalise.pkl"):
        print("⚠️ 'robust_vec_normalise.pkl' not found. Checking models folder...")
        if os.path.exists("models_robust"):
            norm_files = [f for f in os.listdir("models_robust") if f.endswith(".pkl")]
            if norm_files:
                norm_path = os.path.join("models_robust", norm_files[-1])
                print(f"🔄 Found normalization file: {norm_path}")
                env = VecNormalize.load(norm_path, env)
            else:
                print("❌ No normalization file found! Model will likely fail.")
                return
        else:
            print("❌ No normalization file found! Did training finish?")
            return
    else:
        env = VecNormalize.load("robust_vec_normalise.pkl", env)
        print("✅ Normalization stats loaded.")
    
    # Turn OFF training and reward updating
    env.training = False
    env.norm_reward = False
    
    # 4. Load Model
    model_path = "robust_traffic_agent"
    if not os.path.exists(model_path + ".zip"):
        print(f"⚠️ '{model_path}.zip' not found. Checking models folder...")
        if os.path.exists("models_robust"):
            files = [f for f in os.listdir("models_robust") if f.endswith(".zip")]
            if files:
                optimized = [f for f in files if "robust" in f]
                if optimized:
                    latest = max(optimized, key=lambda x: int(x.split('_')[-2].split('.')[0]) if '_' in x else 0)
                else:
                    latest = max(files, key=lambda x: int(x.split('_')[-2].split('.')[0]) if '_' in x else 0)
                model_path = os.path.join("models_robust", latest.replace(".zip", ""))
                print(f"🔄 Found checkpoint: {model_path}")
            else:
                print("❌ No models found! Did training finish?")
                return

    model = PPO.load(model_path)
    print(f"✅ Robust Model Loaded from: {model_path}")

    # 5. Reset and Run
    obs = env.reset()
    done = False
    ambulance_start = 0
    ambulance_end = 0
    ambulance_duration = 0
    step = 0
    vehicle_waiting_times = {}  
    
    print("🚦 Starting Universal Evaluation Run (Random Traffic)...")
    
    try:
        traci.gui.setSchema("View #0", "real world")
    except:
        pass
    
    while not done:
        # 🚨 deterministic=True works safely again because the AI is no longer blind! 🚨
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Break instantly if episode times out
        if done:
            print("\n⚠️ Simulation reached max steps (1000s) before ambulance finished!")
            break
        
        time.sleep(0.05)
        step += 1
        
        # Track Ambulance
        try:
            current_time = traci.simulation.getTime()
            veh_list = traci.vehicle.getIDList()
            
            if step % 20 == 0:
                print(f"   [Debug] Time: {current_time}s | Vehicles on road: {len(veh_list)}")
            
            for veh_id in veh_list:
                if veh_id != "hero_ambulance":
                    waiting = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                    if veh_id not in vehicle_waiting_times or waiting > vehicle_waiting_times[veh_id]:
                        vehicle_waiting_times[veh_id] = waiting

            if "hero_ambulance" in veh_list:
                if ambulance_start == 0:
                    ambulance_start = current_time
                    print(f"🚑 Ambulance entered at: {ambulance_start}")
            
            if ambulance_start > 0 and "hero_ambulance" not in veh_list and ambulance_end == 0:
                ambulance_end = current_time
                ambulance_duration = ambulance_end - ambulance_start
                print(f"🏁 Universal Agent Finished! Total Time: {ambulance_duration} seconds")
                break

        except Exception as e:
            print(f"❌ Error: {e}")
            break

    env.close()
    print("✅ Evaluation Complete.")
    
    if ambulance_start > 0 and ambulance_end == 0:
        ambulance_duration = 999.9  
        print(f"❌ Ambulance failed to complete route. Recorded as {ambulance_duration}s")
    
    if vehicle_waiting_times:
        civilian_avg_wait = sum(vehicle_waiting_times.values()) / len(vehicle_waiting_times)
    else:
        civilian_avg_wait = 0
    
    with open("optimized_result.txt", "w") as f:
        f.write(f"{ambulance_duration}\n")
        f.write(f"{civilian_avg_wait}\n")
        
    print(f"📊 Robust ambulance time: {ambulance_duration}s")
    print(f"📊 Robust civilian avg waiting time: {civilian_avg_wait:.2f}s")
    
    return ambulance_duration, civilian_avg_wait

if __name__ == "__main__":
    test_optimized()