import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sumo_rl
import traci
import os
import time
import glob

def test_latest_checkpoint():
    print("🚀 Scanning for the latest Checkpoint...")
    
    # --- 1. FIND THE LATEST CHECKPOINT ---
    checkpoint_dir = "./modelsops"
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory '{checkpoint_dir}' does not exist!")
        return
        
    # Find all .zip files in the directory
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    
    if not checkpoints:
        print(f"❌ No .zip checkpoints found in '{checkpoint_dir}'!")
        return

    # Function to extract the step number from filenames like "rl_model_optimized_40000_steps.zip"
    def extract_step(filepath):
        filename = os.path.basename(filepath)
        try:
            # Extract all digits from the filename
            numbers = ''.join(filter(str.isdigit, filename))
            return int(numbers) if numbers else -1
        except:
            return -1

    # Get the checkpoint with the highest step number
    latest_checkpoint = max(checkpoints, key=extract_step)
    print(f"🔄 Found Latest Checkpoint: {latest_checkpoint}")

    # --- 2. SETUP THE ENVIRONMENT ---
    # Must perfectly match the training environment (Seed 42) to avoid distribution shift
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
    )
    
    # --- 3. APPLY NORMALIZATION ---
    env = DummyVecEnv([lambda: env])
    
    if not os.path.exists("vec_normalise.pkl"):
        print("❌ 'vec_normalise.pkl' not found! The agent needs this to understand traffic volume.")
        return
    else:
        env = VecNormalize.load("vec_normalise.pkl", env)
        print("✅ Normalization stats loaded.")
    
    # Turn OFF training and reward updating
    env.training = False
    env.norm_reward = False
    
    # --- 4. LOAD THE CHECKPOINT MODEL ---
    model = PPO.load(latest_checkpoint)
    print(f"✅ Checkpoint Brain Successfully Loaded!")

    # --- 5. RUN EVALUATION ---
    obs = env.reset()
    done = False
    ambulance_start = 0
    ambulance_end = 0
    ambulance_duration = 0
    step = 0
    vehicle_waiting_times = {}  
    
    print("🚦 Starting Checkpoint Evaluation Run...")
    
    try:
        traci.gui.setSchema("View #0", "real world")
    except:
        pass
    
    while not done:
        # Use deterministic=True for the most confident, stable actions
        action, _ = model.predict(obs, deterministic=True) 
        obs, reward, done, info = env.step(action)
        
        if done:
            print("\n⚠️ Simulation reached max steps (1000s) before ambulance finished!")
            break
        
        time.sleep(0.05) # Slow down visualization
        step += 1
        
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
                print(f"🏁 Checkpoint Agent Cleared Intersection! Total Time: {ambulance_duration} seconds")
                break

        except Exception as e:
            print(f"❌ Error: {e}")
            break

    env.close()
    
    if ambulance_start > 0 and ambulance_end == 0:
        ambulance_duration = 999.9  
        print(f"❌ Ambulance failed to complete route.")
    
    civilian_avg_wait = sum(vehicle_waiting_times.values()) / len(vehicle_waiting_times) if vehicle_waiting_times else 0
        
    print(f"📊 Checkpoint ambulance time: {ambulance_duration}s")
    print(f"📊 Checkpoint civilian avg waiting time: {civilian_avg_wait:.2f}s")
    
if __name__ == "__main__":
    test_latest_checkpoint()