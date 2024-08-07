import GymWrapper as gw
import time
import HyperparamTuning as ht  # Module for hyperparameter tuning
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from stable_baselines3 import DQN, DDPG, PPO
from log_SimPy import *
from log_RL import *


# Function to build the model based on the specified reinforcement learning algorithm
def build_model():
    model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME *
                4, learning_rate=0.0001, batch_size=20)
    return model


# Experiment conditions
DemandScen = [
    {"min": 8, "max": 15},
    {"min": 5, "max": 12},
    {"min": 11, "max": 18}]
UseCorrection = [True, False]
DailyChange = [1, 0]
Intransit = [1, 0]

# Start timing the computation
start_time = time.time()

for demand in DemandScen:
    for correction in UseCorrection:
        for daily_change in DailyChange:
            for intransit in Intransit:
                USE_CORRECTION, INTRANSIT, DAILY_CHANGE, DEMAND = CREATE_EXPERIMENT_CONDITIONS(
                    correction, intransit, daily_change, demand)

                # Create environment
                env = GymInterface()

                # Set the experiment conditions
                env.scenario = {"Dist_Type": DIST_TYPE,
                                "min": demand["min"], "max": demand["max"]}

                print("START=============================")
                print("State_Demand: UNIF (",
                      env.scenario["min"], ", ", env.scenario["max"], ")")
                print("USE_CORRECTION: ", correction)
                print("DailyChange: ", daily_change)
                print("Intransit: ", intransit)
                print("Initial_Obs: ", env.reset())
                model = build_model()
                # Train the model
                model.learn(total_timesteps=SIM_TIME * N_EPISODES)
                if SAVE_MODEL:
                    model.save(os.path.join(
                        SAVED_MODEL_PATH, SAVED_MODEL_NAME))
                    print(f"{SAVED_MODEL_NAME} is saved successfully")


'''
# Evaluate the trained model
mean_reward, std_reward = gw.evaluate_model(model, env, N_EVAL_EPISODES)
print(
    f"Mean reward over {N_EVAL_EPISODES} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
'''
# Calculate computation time and print it
end_time = time.time()
print(f"Computation time: {(end_time - start_time)/60:.2f} minutes \n")
