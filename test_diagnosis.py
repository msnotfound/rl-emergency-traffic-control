import gymnasium as gym
from stable_baselines3 import PPO
import sumo_rl
import traci
import os

def test_diagnosis():
    print("🚀 Starting Diagnostic Run...")
    
    # 1. Setup Environment
    env = sumo_rl.SumoEnvironment(
        net_file="draft02.net.xml",
        route_file="vtypes.rou.xml,draft02.rou.xml,ambulance.rou.xml",
        out_csv_name="test_results",
        use_gui=True,              
        num_seconds=600,
        fixed_ts=False,            
        single_agent=True
    )

    # 2. Load Agent
    model_path = "my_traffic_agent"
    if not os.path.exists(model_path + ".zip"):
        # Fallback search
        if os.path.exists("models"):
            files = [f for f in os.listdir("models") if f.endswith(".zip")]
            if files:
                latest = max(files, key=lambda x: int(x.split('_')[2])) 
                model_path = os.path.join("models", latest.replace(".zip", ""))
    
    model = PPO.load(model_path)
    print(f"✅ Loaded Model: {model_path}")

    # 3. Reset
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    
    done = False
    step = 0
    
    print("🚦 DIAGNOSIS MODE: Monitoring Ambulance Spawn...")
    
    try:
        traci.gui.setSchema("View #0", "real world")
    except:
        pass

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        
        step += 1
        
        # --- DIAGNOSTIC CHECKS ---
        try:
            current_time = traci.simulation.getTime()
            
            # Check for Ambulance specifically
            ids = traci.vehicle.getIDList()
            if "hero_ambulance" in ids:
                print(f"✅ SUCCESS: Ambulance found at Time {current_time}!")
                break # We found it, diagnosis complete.

            # If it's time for the ambulance (120s) but it's not here...
            if current_time >= 120 and step % 10 == 0:
                # Check the Pending Queue (Vehicles waiting to enter)
                pending_count = traci.simulation.getMinExpectedNumber() - len(ids)
                
                print(f"⚠️ Time {current_time}: Ambulance MISSING.")
                print(f"   - Vehicles on road: {len(ids)}")
                print(f"   - Vehicles waiting in queue: {pending_count}")
                
                # Check if the entry lane is blocked
                # The ambulance starts on "-E2". Let's check the jam length there.
                jam_len = traci.edge.getLastStepHaltingNumber("-E2")
                print(f"   - Jam on entry edge '-E2': {jam_len} cars stopped.")
                
        except Exception as e:
            print(f"Error: {e}")

    env.close()

if __name__ == "__main__":
    test_diagnosis()