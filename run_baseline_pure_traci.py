import traci
import os
import sys
import time

# Check if SUMO_HOME is set (standard safety check)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def run_pure_baseline():
    print("🚀 Starting Pure TraCI Baseline...")
    
    # 1. Define the command to start SUMO
    # We load the config file directly, which already lists your network and routes
    sumoBinary = "sumo-gui" # Use "sumo" for headless
    sumoCmd = [sumoBinary, "-c", "draft02.sumocfg", "--start"]

    # 2. Start the simulation
    traci.start(sumoCmd)
    
    # 3. Setup tracking variables
    ambulance_start = 0
    ambulance_end = 0
    ambulance_duration = 0
    step = 0
    vehicle_waiting_times = {}  # Track max waiting time per vehicle
    
    # 4. Set the GUI to look nice (Optional)
    try:
        traci.gui.setSchema("View #0", "real world")
    except:
        pass

    print("🚦 Simulation Running... (Look at the GUI window)")

    # 5. The Main Loop
    # Run until time 600 OR until all cars are gone
    while step < 1000:
        traci.simulationStep() # Move one step forward
        step += 1
        
        # Slow down slightly so you can see it
        # time.sleep(0.10) 

        # Track the Ambulance and civilian waiting times
        # We wrap this in try-catch to prevent crashes if TraCI hiccups
        try:
            vehicle_list = traci.vehicle.getIDList()
            
            # Track max waiting time for each civilian vehicle
            for veh_id in vehicle_list:
                if veh_id != "hero_ambulance":
                    waiting = traci.vehicle.getWaitingTime(veh_id)
                    # Keep the maximum waiting time seen for this vehicle
                    if veh_id not in vehicle_waiting_times or waiting > vehicle_waiting_times[veh_id]:
                        vehicle_waiting_times[veh_id] = waiting
            
            if "hero_ambulance" in vehicle_list:
                if ambulance_start == 0:
                    ambulance_start = traci.simulation.getTime()
                    print(f"🚑 Ambulance entered at time: {ambulance_start}")
            
            # Check if it finished
            if ambulance_start > 0 and "hero_ambulance" not in vehicle_list and ambulance_end == 0:
                ambulance_end = traci.simulation.getTime()
                ambulance_duration = ambulance_end - ambulance_start
                print(f"🏁 Ambulance FINISHED! Total Time: {ambulance_duration} seconds")
                # Stop immediately after ambulance finishes for fair comparison
                break 
                
        except Exception as e:
            print(f"⚠️ Error checking vehicle: {e}")

    # 6. Clean up
    print("✅ Simulation Finished.")
    traci.close()
    
    # 7. Calculate civilian average waiting time
    if vehicle_waiting_times:
        civilian_avg_wait = sum(vehicle_waiting_times.values()) / len(vehicle_waiting_times)
    else:
        civilian_avg_wait = 0
    
    # 8. Save results to file for plotting
    with open("baseline_result.txt", "w") as f:
        f.write(f"{ambulance_duration}\n")
        f.write(f"{civilian_avg_wait}\n")
    print(f"📊 Baseline ambulance time: {ambulance_duration}s")
    print(f"📊 Baseline civilian avg waiting time: {civilian_avg_wait:.2f}s")
    
    return ambulance_duration, civilian_avg_wait

if __name__ == "__main__":
    run_pure_baseline()