from InventoryMgtEnv import GymInterface
from config_SimPy import *
from config_RL import *
import numpy as np
import HyperparamTuning as ht  # Module for hyperparameter tuning
import time
from stable_baselines3 import DQN, DDPG, PPO
import visualization
from log import *
import environment as Env
import pandas as pd

# Create environment
env = GymInterface()

# Function to evaluate the trained model


def evaluate_model(model, env, num_episodes):
    all_rewards = []  # List to store total rewards for each episode
    XAI = []  # List for storing data for explainable AI purposes
    test_order_mean = []  # List to store average orders per episode
    for i in range(num_episodes):
        ORDER_HISTORY.clear()  # Clear order history at the start of each episode
        DAILY_REPORTS.clear()  # Clear daily reports at the start of each episode
        obs = env.reset()  # Reset the environment to get initial observation
        episode_reward = 0  # Initialize reward for the episode
        done = False  # Flag to check if episode is finished

        while not done:
            action, _ = model.predict(obs)  # Get action from model
            # Execute action in environment
            obs, reward, done, _ = env.step(action)
            episode_reward += reward  # Accumulate rewards
            XAI.append(
                [_ for _ in list(env.cap_current_state())])
            XAI[-1].append(action)  # Append action to XAI data
            ORDER_HISTORY.append(action[0])  # Log order history
        all_rewards.append(episode_reward)  # Store total reward for episode
        report(env, i)  # Visualize the environment

        # Calculate mean order for the episode
        test_order_mean.append(sum(ORDER_HISTORY) / len(ORDER_HISTORY))
    print("Order_Average:", test_order_mean)
    if XAI_TRAIN_EXPORT:
        df = pd.DataFrame(XAI)  # Create a DataFrame from XAI data
        df.to_csv(f"{XAI_TRAIN}/XAI_DATA.csv")  # Save XAI data to CSV file

    # Calculate mean reward across all episodes
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)  # Calculate standard deviation of rewards

    return mean_reward, std_reward  # Return mean and std of rewards

# Function to visualize the environment


def report(env, i):
    print(len(DAILY_REPORTS))
    export_Daily_Report = []
    for x in range(len(env.inventoryList)):
        for report in DAILY_REPORTS:
            export_Daily_Report.append(report[x])
    if VISUALIAZTION.count(1) > 0:
        visualization.visualization(export_Daily_Report, i)

    if DAILY_REPORT_EXPORT:
        daily_reports = pd.DataFrame(export_Daily_Report)
        daily_reports.columns = ["Day", "Name", "Type",
                                 "Start", "Income", "Outcome", "End"]
        daily_reports.to_csv(os.path.join(REPORT_LOGS, f'Test_{i}.csv'))

# Function to build the model based on the specified reinforcement learning algorithm


def build_model():
    if RL_ALGORITHM == "DQN":
        # model = DQN("MlpPolicy", env, verbose=0)
        model = DQN("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
                    batch_size=BEST_PARAMS['batch_size'], verbose=0)
    elif RL_ALGORITHM == "DDPG":
        model = DDPG("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
                     batch_size=BEST_PARAMS['batch_size'], verbose=0)
    elif RL_ALGORITHM == "PPO":
        model = PPO("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
                    batch_size=BEST_PARAMS['batch_size'], verbose=0)
        print(env.observation_space)
    return model


'''
def export_report():
    for x in range(len(inventoryList)):
        for report in DAILY_REPORTS:
            export_Daily_Report.append(report[x])
    daily_reports = pd.DataFrame(export_Daily_Report)
    daily_reports.columns = ["Day", "Name", "Type",
                         "Start", "Income", "Outcome", "End"]
    daily_reports.to_csv("./Daily_Report.csv")
'''


# Start timing the computation
start_time = time.time()

# Run hyperparameter optimization if enabled
if OPTIMIZE_HYPERPARAMETERS:
    ht.run_optuna(env)

# Build the model
model = build_model()
# Train the model
model.learn(total_timesteps=SIM_TIME * N_EPISODES)
# Optionally render the environment
env.render()

# Evaluate the trained model
mean_reward, std_reward = evaluate_model(model, env, N_EVAL_EPISODES)
print(
    f"Mean reward over {N_EVAL_EPISODES} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")


# Calculate computation time and print it
end_time = time.time()
print(f"Computation time: {(end_time - start_time)/3600:.2f} hours")
