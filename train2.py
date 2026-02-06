import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import sumo_rl
import traci  # <--- Essential for custom reward
import os

def custom_ambulance_reward(traffic_signal):
    """
    Reward = -(Civilian Wait) - (Ambulance Wait * Weight)
    """
    # --------------------------------------------------------------------------
    # FIX IS HERE: We MUST use sum() to convert the list of lane times into one number
    # --------------------------------------------------------------------------
    lane_waits = traffic_signal.get_accumulated_waiting_time_per_lane()
    civilian_penalty = sum(lane_waits) 
    
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
    """
    Runs a single evaluation episode to see how fast the ambulance moves.
    """
    print("🚑 Starting Evaluation Run...")
    obs = env.reset()
    done = False
    ambulance_start_time = 0
    ambulance_end_time = 0
    
    while not done:
        action, _ = model.predict(obs)
        
        # Unpack 4 values (for your older gym version)
        obs, reward, done, info = env.step(action)
        
        # Track ambulance
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

    # Define the Checkpoint: Save every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="rl_model"
    )

    env = sumo_rl.SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        fixed_ts=False,  # We want AI control now
        out_csv_name=None,  # Disable CSV output
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
    model.learn(total_timesteps=10000, callback=checkpoint_callback)

    model.save("my_traffic_agent")
    print("Model saved successfully.")
    
    # Run evaluation
    evaluate_ambulance_performance(env, model)

if __name__ == "__main__":
    train()