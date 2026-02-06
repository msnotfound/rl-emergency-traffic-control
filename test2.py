import gymnasium as gym
from stable_baselines3 import PPO
import sumo_rl
import traci
import os

def test_model():
    print("🚀 Loading Trained Model...")
    
    # 1. Setup Environment
    env = sumo_rl.SumoEnvironment(
        net_file="draft02.net.xml",
        route_file="vtypes.rou.xml,draft02.rou.xml,ambulance.rou.xml",
        out_csv_name="test_results",
        use_gui=True,              
        num_seconds=600,
        fixed_ts=False,            
        yellow_time=4,
        min_green=5,
        max_green=60,
        single_agent=True
    )

    # 2. Load Agent
    model_path = "my_traffic_agent"
    if not os.path.exists(model_path + ".zip"):
        print(f"⚠️ '{model_path}.zip' not found. Checking models folder...")
        if os.path.exists("models"):
            files = [f for f in os.listdir("models") if f.endswith(".zip")]
            if files:
                latest = max(files, key=lambda x: int(x.split('_')[2])) 
                model_path = os.path.join("models", latest.replace(".zip", ""))
                print(f"🔄 Found checkpoint: {model_path}")
            else:
                print("❌ No models found! Did training finish?")
                return

    model = PPO.load(model_path)
    print(f"✅ Model loaded from: {model_path}")

    # 3. Reset
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs = reset_result[0]
    else:
        obs = reset_result
    
    done = False
    ambulance_start = 0
    ambulance_end = 0
    step = 0
    
    print("🚦 Starting DEBUG Evaluation Run...")
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        # 4. Step
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        
        # Slow down visualization so you can see the cars
        # import time
        # time.sleep(0.1)
        
        step += 1
        
        # --- DEBUGGING BLOCK ---
        try:
            current_time = traci.simulation.getTime()
            veh_list = traci.vehicle.getIDList()
            
            # Print status every 10 seconds so you know it's alive
            if step % 10 == 0:
                print(f"   [Debug] Time: {current_time}s | Vehicles on road: {len(veh_list)}")

            # Check SPECIFICALLY for the ambulance
            if "hero_ambulance" in veh_list:
                print(f"🚨 SAW AMBULANCE at Time {current_time}!")
                if ambulance_start == 0:
                    ambulance_start = current_time
                    print(f"   --> Recording Start Time: {ambulance_start}")
            
            # Check if it finished
            if ambulance_start > 0 and "hero_ambulance" not in veh_list and ambulance_end == 0:
                ambulance_end = current_time
                duration = ambulance_end - ambulance_start
                print(f"🏁 AI Agent Finished! Total Time: {duration} seconds")
                # break # Stop if you want to exit immediately

        except Exception as e:
            print(f"❌ CRITICAL ERROR in loop: {e}")
            break

    env.close()

if __name__ == "__main__":
    test_model()