import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO
import sumo_rl  # <--- FIXED TYPO (was sumot_rl)
import traci    # <--- Added global import for safe access
import os

# def custom_ambulance_reward(traffic_signal):
#     """
#     Reward = -(Civilian Wait) - (Ambulance Wait * Weight)
#     """
#     # 1. Calculate Civilian Penalty (Standard)
#     civilian_penalty = sum(traffic_signal.get_accumulated_waiting_time_per_lane())
    
#     # 2. Calculate Emergency Penalty
#     ambulance_penalty = 0
    
#     # FIX: Use global 'traci' instead of 'traffic_signal.env.traci_connection'
#     vehicle_list = traci.vehicle.getIDList()
    
#     for veh_id in vehicle_list:
#         if traci.vehicle.getTypeID(veh_id) == "ambulance_type":
#             speed = traci.vehicle.getSpeed(veh_id)
#             if speed < 1.0:
#                 # MASSIVE PENALTY for stopping the ambulance
#                 ambulance_penalty += 1000 
#                 # print(f"⚠️ AMBULANCE IS STUCK! Applying massive penalty.")

#     reward = -1 * (civilian_penalty + ambulance_penalty)
#     return reward
def custom_ambulance_reward(traffic_signal):
    """
    Reward = -(Civilian Wait) - (Ambulance Wait * Weight)
    """
    # FIX: Add sum() here to turn the list [10.0, 5.0] into the number 15.0
    civilian_penalty = sum(traffic_signal.get_accumulated_waiting_time_per_lane())
    
    # 2. Calculate Emergency Penalty
    ambulance_penalty = 0
    
    # Use global 'traci' safely
    try:
        vehicle_list = traci.vehicle.getIDList()
        for veh_id in vehicle_list:
            if traci.vehicle.getTypeID(veh_id) == "ambulance_type":
                speed = traci.vehicle.getSpeed(veh_id)
                # If ambulance is moving slower than 1 m/s, apply penalty
                if speed < 1.0:
                    ambulance_penalty += 1000 
    except:
        pass

    # Now both are numbers, so they can be added
    reward = -1 * (civilian_penalty + ambulance_penalty)
    return reward

def evaluate_ambulance_performance(env, model):
    obs = env.reset()
    done = False
    ambulance_start_time = 0
    ambulance_end_time = 0
    
    print("🚑 Starting Evaluation Run...")
    
    while not done:
        action, _ = model.predict(obs)
        
        # FIX: Unpack only 4 values (for your version)
        obs, reward, done, info = env.step(action)
        
        # FIX: Use global 'traci'
        if "hero_ambulance" in traci.vehicle.getIDList():
            if ambulance_start_time == 0:
                ambulance_start_time = traci.simulation.getTime()
                print("🚑 Ambulance entered at:", ambulance_start_time)
        
        if ambulance_start_time > 0 and "hero_ambulance" not in traci.vehicle.getIDList() and ambulance_end_time == 0:
            ambulance_end_time = traci.simulation.getTime()
            travel_time = ambulance_end_time - ambulance_start_time
            print(f"🏁 Ambulance finished! Total Travel Time: {travel_time} seconds")

def train():
    net_file = "draft02.net.xml"
    route_file = "vtypes.rou.xml,draft02.rou.xml,ambulance.rou.xml"

    env = sumo_rl.SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        fixed_ts=False,  # We want AI control now
        out_csv_name="traffic_result",
        use_gui=False,   # Keep False for fast training
        num_seconds=600,
        yellow_time=4,
        min_green=5,
        max_green=60,
        single_agent=True,
        reward_fn=custom_ambulance_reward
    )

    print("Initializing Agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        batch_size=256
    )

    print("Starting training... (This might take a while)")
    model.learn(total_timesteps=100000)

    model.save("my_traffic_agent")
    print("Model saved successfully.")
    
    # Run evaluation (make sure to close env first to be safe, or just run it)
    # Re-enabling GUI for evaluation requires a reset or new env, 
    # but for now we just run it on the existing headless env to get the time.
    evaluate_ambulance_performance(env, model)

if __name__ == "__main__":
    train()