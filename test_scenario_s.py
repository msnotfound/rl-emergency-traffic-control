import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sumo_rl
import traci
import os
import time
import json

def test_scenario_s():
    print("🚀 Loading S-Scenario Trained Model...")
    
    # 1. Setup Same Environment (controls all traffic signals: J4 and J6)
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
        single_agent=True  # Single agent controlling all traffic signals
    )
    
    # 2. Re-Apply Normalization Wrapper
    env = DummyVecEnv([lambda: env])
    
    # Check if normalization file exists
    norm_path = "models_s/vec_normalize_s.pkl"
    if not os.path.exists(norm_path):
        print(f"⚠️ '{norm_path}' not found. Checking for alternatives...")
        if os.path.exists("models_s"):
            norm_files = [f for f in os.listdir("models_s") if f.endswith("_vecnormalize.pkl")]
            if norm_files:
                norm_path = os.path.join("models_s", norm_files[-1])
                print(f"🔄 Found normalization file: {norm_path}")
            else:
                print("❌ No normalization file found! Model will likely fail.")
                return
        else:
            print("❌ models_s/ directory not found! Did training finish?")
            return
    
    env = VecNormalize.load(norm_path, env)
    
    # Turn OFF training and reward updating
    env.training = False
    env.norm_reward = False
    
    # 3. Load Model
    model_path = "models_s/optimized_traffic_agent_s"
    if not os.path.exists(model_path + ".zip"):
        print(f"⚠️ '{model_path}.zip' not found. Checking for checkpoints...")
        if os.path.exists("models_s"):
            files = [f for f in os.listdir("models_s") if f.endswith(".zip")]
            if files:
                # Find the latest checkpoint
                scenario_files = [f for f in files if "scenario_s" in f]
                if scenario_files:
                    latest = max(scenario_files, key=lambda x: int(x.split('_')[-2]) if x.split('_')[-2].isdigit() else 0)
                else:
                    latest = max(files, key=lambda x: int(x.split('_')[-2]) if x.split('_')[-2].isdigit() else 0)
                model_path = os.path.join("models_s", latest.replace(".zip", ""))
                print(f"🔄 Found checkpoint: {model_path}")
            else:
                print("❌ No models found! Did training finish?")
                return

    model = PPO.load(model_path)
    print(f"✅ S-Scenario Model Loaded from: {model_path}")

    # 4. Reset and Run
    obs = env.reset()
    done = False
    ambulance_start = 0
    ambulance_end = 0
    ambulance_duration = 0
    step = 0
    vehicle_waiting_times = {}  # Track max waiting time per vehicle
    
    # Enhanced tracking for S-scenario
    ambulance_at_j4 = 0
    ambulance_at_j6 = 0
    j4_crossing_time = 0
    j6_crossing_time = 0
    
    print("🚦 Starting S-Scenario Evaluation Run...")
    print("   - Tracking ambulance through S-route (J4 → E3 → J6)")
    
    # Set GUI view
    try:
        traci.gui.setSchema("View #0", "real world")
    except:
        pass
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        step += 1
        time.sleep(0.0)  # Slow down for visualization
        # Track Ambulance and metrics
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
                    if veh_id not in vehicle_waiting_times or waiting > vehicle_waiting_times[veh_id]:
                        vehicle_waiting_times[veh_id] = waiting

            # Check for ambulance and track its route
            if "hero_ambulance" in veh_list:
                if ambulance_start == 0:
                    ambulance_start = current_time
                    print(f"🚑 Ambulance entered at: {ambulance_start}s (starting S-route)")
                
                # Track ambulance position along S-route
                amb_road = traci.vehicle.getRoadID("hero_ambulance")
                
                if "E3" in amb_road and ambulance_at_j4 == 0:
                    ambulance_at_j4 = current_time
                    j4_crossing_time = ambulance_at_j4 - ambulance_start
                    print(f"   ✓ Ambulance crossed J4 at {ambulance_at_j4}s (took {j4_crossing_time:.1f}s)")
                
                if "E5" in amb_road and ambulance_at_j6 == 0:
                    ambulance_at_j6 = current_time
                    j6_crossing_time = ambulance_at_j6 - ambulance_at_j4 if ambulance_at_j4 > 0 else 0
                    print(f"   ✓ Ambulance crossed J6 at {ambulance_at_j6}s (took {j6_crossing_time:.1f}s)")
            
            # Check if finished
            if ambulance_start > 0 and "hero_ambulance" not in veh_list and ambulance_end == 0:
                ambulance_end = current_time
                ambulance_duration = ambulance_end - ambulance_start
                print(f"🏁 S-Scenario Complete! Total Ambulance Time: {ambulance_duration}s")
                break

        except Exception as e:
            print(f"❌ Error: {e}")
            break

    env.close()
    print("✅ S-Scenario Evaluation Complete.")
    
    # Calculate civilian average waiting time
    if vehicle_waiting_times:
        civilian_avg_wait = sum(vehicle_waiting_times.values()) / len(vehicle_waiting_times)
    else:
        civilian_avg_wait = 0
    
    # Save detailed results to file for plotting
    results = {
        "ambulance_total_time": ambulance_duration,
        "civilian_avg_wait": civilian_avg_wait,
        "j4_crossing_time": j4_crossing_time,
        "j6_crossing_time": j6_crossing_time,
        "total_vehicles": len(vehicle_waiting_times)
    }
    
    with open("optimized_result_s.txt", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 S-Scenario Results:")
    print(f"   - Total ambulance time: {ambulance_duration}s")
    print(f"   - J4 crossing time: {j4_crossing_time:.1f}s")
    print(f"   - J6 crossing time: {j6_crossing_time:.1f}s")
    print(f"   - Civilian avg waiting time: {civilian_avg_wait:.2f}s")
    print(f"   - Total vehicles processed: {len(vehicle_waiting_times)}")
    
    return results

if __name__ == "__main__":
    test_scenario_s()
