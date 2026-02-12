import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sumo_rl
import traci
import os
import time

def test_optimized():
    print("🚀 Loading Optimized Trained Model...")
    
    # 1. Setup Same Environment
    env = sumo_rl.SumoEnvironment(
        net_file="draft02.net.xml",
        route_file="vtypes.rou.xml,draft02.rou.xml,ambulance.rou.xml",
        out_csv_name=None,
        use_gui=True,
        num_seconds=1000,
        fixed_ts=False,
        yellow_time=4,
        min_green=5,
        max_green=60,
        single_agent=True
    )
    
    # 2. Re-Apply Normalization Wrapper
    env = DummyVecEnv([lambda: env])
    
    # Check if normalization file exists
    if not os.path.exists("vec_normalize.pkl"):
        print("⚠️ 'vec_normalize.pkl' not found. Checking models folder...")
        if os.path.exists("modelsop"):
            # Try to find a normalization file in models/
            norm_files = [f for f in os.listdir("modelsop") if f.endswith("_vecnormalize.pkl")]
            if norm_files:
                norm_path = os.path.join("modelsop", norm_files[-1])
                print(f"🔄 Found normalization file: {norm_path}")
                env = VecNormalize.load(norm_path, env)
            else:
                print("❌ No normalization file found! Model will likely fail.")
                return
        else:
            print("❌ No normalization file found! Did training finish?")
            return
    else:
        # Load the stats we learned during training
        env = VecNormalize.load("vec_normalize.pkl", env)
    
    # Turn OFF training and reward updating (we just want to test now)
    env.training = False
    env.norm_reward = False
    
    # 3. Load Model
    model_path = "optimized_traffic_agent"
    if not os.path.exists(model_path + ".zip"):
        print(f"⚠️ '{model_path}.zip' not found. Checking models folder...")
        if os.path.exists("modelsop"):
            files = [f for f in os.listdir("modelsop") if f.endswith(".zip")]
            if files:
                # Find the optimized model or latest
                optimized = [f for f in files if "optimized" in f]
                if optimized:
                    latest = max(optimized, key=lambda x: int(x.split('_')[-2].split('.')[0]))
                else:
                    latest = max(files, key=lambda x: int(x.split('_')[-2].split('.')[0]))
                model_path = os.path.join("modelsop", latest.replace(".zip", ""))
                print(f"🔄 Found checkpoint: {model_path}")
            else:
                print("❌ No models found! Did training finish?")
                return

    model = PPO.load(model_path)
    print(f"✅ Optimized Model Loaded from: {model_path}")

    # 4. Reset and Run
    obs = env.reset()
    done = False
    ambulance_start = 0
    ambulance_end = 0
    ambulance_duration = 0
    step = 0
    vehicle_waiting_times = {}  # Track max waiting time per vehicle
    
    print("🚦 Starting Optimized Evaluation Run...")
    
    # Set GUI view
    try:
        traci.gui.setSchema("View #0", "real world")
    except:
        pass
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Slow down visualization so you can see the cars
        # time.sleep(0.05)
        
        step += 1
        
        # Track Ambulance
        try:
            current_time = traci.simulation.getTime()
            veh_list = traci.vehicle.getIDList()
            
            # Print status every 20 steps
            if step % 20 == 0:
                print(f"   [Debug] Time: {current_time}s | Vehicles on road: {len(veh_list)}")
            
            # Track max waiting time for each civilian vehicle
            for veh_id in veh_list:
                if veh_id != "hero_ambulance":
                    waiting = traci.vehicle.getWaitingTime(veh_id)
                    # Keep the maximum waiting time seen for this vehicle
                    if veh_id not in vehicle_waiting_times or waiting > vehicle_waiting_times[veh_id]:
                        vehicle_waiting_times[veh_id] = waiting

            # Check for ambulance
            if "hero_ambulance" in veh_list:
                if ambulance_start == 0:
                    ambulance_start = current_time
                    print(f"🚑 Ambulance entered at: {ambulance_start}")
            
            # Check if finished
            if ambulance_start > 0 and "hero_ambulance" not in veh_list and ambulance_end == 0:
                ambulance_end = current_time
                ambulance_duration = ambulance_end - ambulance_start
                print(f"🏁 Optimized Agent Finished! Total Time: {ambulance_duration} seconds")
                # Exit immediately after ambulance finishes to avoid multi-episode data
                break

        except Exception as e:
            print(f"❌ Error: {e}")
            break

    env.close()
    print("✅ Evaluation Complete.")
    
    # Calculate civilian average waiting time
    if vehicle_waiting_times:
        civilian_avg_wait = sum(vehicle_waiting_times.values()) / len(vehicle_waiting_times)
    else:
        civilian_avg_wait = 0
    
    # Save results to file for plotting
    with open("optimized_result.txt", "w") as f:
        f.write(f"{ambulance_duration}\n")
        f.write(f"{civilian_avg_wait}\n")
    print(f"📊 Optimized ambulance time: {ambulance_duration}s")
    print(f"📊 Optimized civilian avg waiting time: {civilian_avg_wait:.2f}s")
    
    return ambulance_duration, civilian_avg_wait

if __name__ == "__main__":
    test_optimized()
