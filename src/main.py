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
    if RL_ALGORITHM == "PPO":
        model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME,learning_rate=0.0003, batch_size=64) #DEFAULT: learning_rate=0.0003, batch_size=64 => 28 mins
        # [Train 2] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME, learning_rate=0.0001, batch_size=16) => 50 mins
        # [Train 3] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME, learning_rate=0.0002, batch_size=16) => 49 mins
        # [Train 4] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME, learning_rate=0.00015, batch_size=20) => 44 mins
        # [Train 5] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME, learning_rate=0.0001, batch_size=20) => 39 mins
        # [Train 6] # => 40 mins
        # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME *
        #             4, learning_rate=0.0001, batch_size=20)
        # [Train 7] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME*2, learning_rate = 0.0001, batch_size = 20) => 36 mins
        # [Train 8] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME*10, learning_rate = 0.0001, batch_size = 20) => 40 mins

        # model = PPO("MlpPolicy", env, learning_rate=BEST_PARAMS['LEARNING_RATE'], gamma=BEST_PARAMS['GAMMA'],
        #             batch_size=BEST_PARAMS['BATCH_SIZE'], n_steps=BEST_PARAMS['N_STEPS'], verbose=0)
        print(env.observation_space)
    elif RL_ALGORITHM == "DQN":
        pass
    elif RL_ALGORITHM == "DDPG":
        pass
    return model


# Start timing the computation
start_time = time.time()

# Create environment
env = GymInterface()

# Run hyperparameter optimization if enabled
if OPTIMIZE_HYPERPARAMETERS:
    ht.run_optuna()
    # Calculate computation time and print it
    end_time = time.time()
    print(f"Computation time: {(end_time - start_time)/60:.2f} minutes ")
else:
    # Build the model
    if LOAD_MODEL:
        if RL_ALGORITHM == "PPO":
            model = PPO.load(os.path.join(
                SAVED_MODEL_PATH, LOAD_MODEL_NAME), env=env)
        print(f"{LOAD_MODEL_NAME} is loaded successfully")
    else:
        print("State_CORRECTION: ", USE_CORRECTION)
        print("Initial_Obs: ", env.reset())
        model = build_model()
        # Train the model
        model.learn(total_timesteps=SIM_TIME * N_EPISODES)
        if SAVE_MODEL:
            model.save(os.path.join(SAVED_MODEL_PATH, SAVED_MODEL_NAME))
            print(f"{SAVED_MODEL_NAME} is saved successfully")

        if STATE_TRAIN_EXPORT:
            gw.export_state('TRAIN')
    training_end_time = time.time()

    # Evaluate the trained model
    mean_reward, std_reward = gw.evaluate_model(model, env, N_EVAL_EPISODES)
    print(
        f"Mean reward over {N_EVAL_EPISODES} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
    # Calculate computation time and print it
    end_time = time.time()
    print(f"Computation time: {(end_time - start_time)/60:.2f} minutes \n",
          f"Training time: {(training_end_time - start_time)/60:.2f} minutes \n ",
          f"Test time:{(end_time - training_end_time)/60:.2f} minutes")


# Optionally render the environment
env.render()
