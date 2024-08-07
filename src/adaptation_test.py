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
    if RL_ALGORITHM == "DQN":
        model = DQN("MlpPolicy", env, verbose=0)
        # model = DQN("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
        #             batch_size=BEST_PARAMS['batch_size'], verbose=0)
    elif RL_ALGORITHM == "DDPG":
        model = DQN("MlpPolicy", env, verbose=0,)
        # model = DDPG("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
        #              batch_size=BEST_PARAMS['batch_size'], verbose=0)
    elif RL_ALGORITHM == "PPO":
        # [Train 1] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME) DEFAULT: learning_rate=0.0003, batch_size=64 => 28 mins
        # [Train 2] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME, learning_rate=0.0001, batch_size=16) => 50 mins
        # [Train 3] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME, learning_rate=0.0002, batch_size=16) => 49 mins
        # [Train 4] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME, learning_rate=0.00015, batch_size=20) => 44 mins
        # [Train 5] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME, learning_rate=0.0001, batch_size=20) => 39 mins
        # [Train 6] # => 40 mins
        model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME *
                    4, learning_rate=0.0001, batch_size=20)
        # [Train 7] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME*2, learning_rate = 0.0001, batch_size = 20) => 36 mins
        # [Train 8] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME*10, learning_rate = 0.0001, batch_size = 20) => 40 mins

        # model = PPO("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
        #             batch_size=BEST_PARAMS['batch_size'], n_steps=SIM_TIME, verbose=0)
        print(env.observation_space)
    return model


# Start timing the computation
start_time = time.time()

# Create environment
env = GymInterface()

# Build the model
model = build_model()
if LOAD_MODEL:
    saved_model = PPO.load(os.path.join(
        SAVED_MODEL_PATH, LOAD_MODEL_NAME), env=env)  # Load the saved model
    # 정책 네트워크의 파라미터 복사
    model.policy.load_state_dict(saved_model.policy.state_dict())

# Train the model
model.learn(total_timesteps=SIM_TIME * N_EPISODES)

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
