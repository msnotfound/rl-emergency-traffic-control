import sumo_rl
import traci

def run_baseline():
    env = sumo_rl.SumoEnvironment(
        net_file="draft02.net.xml",
        route_file="vtypes.rou.xml,draft02.rou.xml,ambulance.rou.xml",
        use_gui=True,
        num_seconds=600,
        fixed_ts=True,
    )

    obs = env.reset()
    done = False
    ambulance_start = 0
    ambulance_end = 0

    print("🚦 Starting Fixed-Time Baseline Simulation...")
    
    # --- FIX: Set the GUI Delay to 100ms so it doesn't vanish instantly ---
    # "View #0" is the default ID of the SUMO window
    try:
        traci.gui.setBound("View #0", "draft02.net.xml") # Optional: centers view
        traci.gui.setSchema("View #0", "real world")     # Makes it look nice
    except:
        pass # Ignore if GUI isn't ready
    
    while not done:
        # Action None = "Do nothing, let the fixed timer run"
        obs, reward, done, info = env.step(None)

        # --- FIX: Slow down manually if setSchema fails ---
        # traci.simulation.getDeltaT() usually returns 1.0s
        import time
        time.sleep(0.05) # Sleep 50ms per step to make it visible
        
        # --- FIX: Print current time so you know it's running ---
        current_time = traci.simulation.getTime()
        if current_time % 10 == 0: # Print every 10 sim-seconds
            print(f"Time Step: {current_time}")

        # Track the Ambulance
        if "hero_ambulance" in traci.vehicle.getIDList():
            if ambulance_start == 0:
                ambulance_start = current_time
                print(f"🚑 Ambulance entered at: {ambulance_start}")
        
        # Check finish
        if ambulance_start > 0 and "hero_ambulance" not in traci.vehicle.getIDList() and ambulance_end == 0:
            ambulance_end = current_time
            print(f"🏁 Ambulance finished (Fixed Time) in: {ambulance_end - ambulance_start} seconds")

    env.close()

if __name__ == "__main__":
    run_baseline()