import traci
import os
import sys
import time
import json

# Check if SUMO_HOME is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def run_baseline_scenario_s():
    print("🚀 Starting S-Scenario Baseline (Fixed-Time Signals)...")
    
    # 1. Define the command to start SUMO
    sumoBinary = "sumo-gui"  # Use "sumo" for headless
    sumoCmd = [sumoBinary, "-c", "draft02_scenario_s.sumocfg", "--start"]

    # 2. Start the simulation
    traci.start(sumoCmd)
    
    # 3. Setup tracking variables
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
    
    # 4. Set the GUI to look nice (Optional)
    try:
        traci.gui.setSchema("View #0", "real world")
    except:
        pass

    print("🚦 S-Scenario Baseline Running...")
    print("   - Fixed-time traffic signals")
    print("   - S-shaped ambulance route (J4 → E3 → J6)")
    print("   - Heavy traffic (300+ vehicles)")

    # 5. The Main Loop
    while step < 1000:
        traci.simulationStep()
        step += 1
        time.sleep(.050)

        # Track the Ambulance and civilian waiting times
        try:
            current_time = traci.simulation.getTime()
            vehicle_list = traci.vehicle.getIDList()
            
            # Track max waiting time for each civilian vehicle
            for veh_id in vehicle_list:
                if veh_id != "hero_ambulance":
                    waiting = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                    if veh_id not in vehicle_waiting_times or waiting > vehicle_waiting_times[veh_id]:
                        vehicle_waiting_times[veh_id] = waiting
            
            # Check for ambulance and track its route
            if "hero_ambulance" in vehicle_list:
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
            if ambulance_start > 0 and "hero_ambulance" not in vehicle_list and ambulance_end == 0:
                ambulance_end = current_time
                ambulance_duration = ambulance_end - ambulance_start
                print(f"🏁 Baseline U-Scenario Complete! Total Time: {ambulance_duration}s")
                break
                
        except Exception as e:
            print(f"⚠️ Error checking vehicle: {e}")

    # 6. Clean up
    print("✅ Baseline Simulation Finished.")
    traci.close()
    
    # 7. Calculate civilian average waiting time
    if vehicle_waiting_times:
        civilian_avg_wait = sum(vehicle_waiting_times.values()) / len(vehicle_waiting_times)
    else:
        civilian_avg_wait = 0
    
    # 8. Save detailed results to file for plotting
    results = {
        "ambulance_total_time": ambulance_duration,
        "civilian_avg_wait": civilian_avg_wait,
        "j4_crossing_time": j4_crossing_time,
        "j6_crossing_time": j6_crossing_time,
        "total_vehicles": len(vehicle_waiting_times)
    }
    
    with open("baseline_result_s.txt", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Baseline S-Scenario Results:")
    print(f"   - Total ambulance time: {ambulance_duration}s")
    print(f"   - J4 crossing time: {j4_crossing_time:.1f}s")
    print(f"   - J6 crossing time: {j6_crossing_time:.1f}s")
    print(f"   - Civilian avg waiting time: {civilian_avg_wait:.2f}s")
    print(f"   - Total vehicles processed: {len(vehicle_waiting_times)}")
    
    return results

if __name__ == "__main__":
    run_baseline_scenario_s()
