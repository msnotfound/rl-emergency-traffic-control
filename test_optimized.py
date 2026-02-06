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
    step = 0
    
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
        time.sleep(0.05)
        
        step += 1
        
        # Track Ambulance
        try:
            current_time = traci.simulation.getTime()
            veh_list = traci.vehicle.getIDList()
            
            # Print status every 20 steps
            if step % 20 == 0:
                print(f"   [Debug] Time: {current_time}s | Vehicles on road: {len(veh_list)}")

            # Check for ambulance
            if "hero_ambulance" in veh_list:
                if ambulance_start == 0:
                    ambulance_start = current_time
                    print(f"🚑 Ambulance entered at: {ambulance_start}")
            
            # Check if finished
            if ambulance_start > 0 and "hero_ambulance" not in veh_list and ambulance_end == 0:
                ambulance_end = current_time
                duration = ambulance_end - ambulance_start
                print(f"🏁 Optimized Agent Finished! Total Time: {duration} seconds")
                # Uncomment to exit immediately after ambulance finishes
                # break

        except Exception as e:
            print(f"❌ Error: {e}")
            break

    env.close()
    print("✅ Evaluation Complete.")

if __name__ == "__main__":
    test_optimized()
