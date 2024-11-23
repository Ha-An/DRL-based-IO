import GymWrapper as gw
import time
import HyperparamTuning as ht  # Module for hyperparameter tuning
from GymWrapper import GymInterface  # Custom interface for Gym environments
from config_SimPy import *  # Import configuration settings for simulation
from config_RL import *  # Import configuration settings for reinforcement learning
from stable_baselines3 import PPO  # Import PPO algorithm from Stable Baselines3
from log_SimPy import *  # Import logging utilities for simulation
from log_RL import *  # Import logging utilities for reinforcement learning
# For logging data to TensorBoard
from torch.utils.tensorboard import SummaryWriter
import pandas as pd  # For data manipulation and analysis
# Evaluation callback for periodic performance testing
from stable_baselines3.common.callbacks import EvalCallback

# Dictionary to store the results of each experiment
experiment_result = {}

# Define paths for saving models
current_dir = os.path.dirname(__file__)  # Current directory path
parent_dir = os.path.dirname(current_dir)  # Parent directory path
# Path for saving/loading a meta-trained model
meta_model_path = os.path.join(SAVED_MODEL_PATH, 'Adapted_Model')

# Custom callback class to save rewards from each evaluation step


class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super(CustomEvalCallback, self).__init__(*args, **kwargs)
        self.rewards = []  # List to store the mean rewards from each evaluation

    def _on_step(self) -> bool:
        # Perform evaluation step and store the mean reward at each evaluation frequency
        result = super(CustomEvalCallback, self)._on_step()

        if self.n_calls % self.eval_freq == 0:  # Check if it's time for evaluation
            # Store the last evaluation's mean reward
            self.rewards.append(self.last_mean_reward)

        return result  # Continue with training

# Function to create and return an evaluation callback


def make_call_back(env):
    eval_callback = CustomEvalCallback(
        env,                     # Evaluation environment
        # Evaluation frequency (in simulation time steps)
        eval_freq=SIM_TIME * 1000,
        n_eval_episodes=15,           # Number of evaluation episodes to average reward over
        log_path='./logs/',           # Path for saving evaluation logs
        best_model_save_path='./logs/',  # Path for saving the best-performing model
        deterministic=True,           # Whether to use deterministic actions during evaluation
        render=False                  # Render environment during evaluation if set to True
    )
    return eval_callback

# Function to build and return a new PPO model with specific hyperparameters


def build_model(env):
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0001,
                batch_size=20, n_steps=SIM_TIME)  # Initialize PPO with given settings
    return model

# Function to load an existing meta-trained model for adaptation to new environments


def load_model(env):
    meta_model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0001,
                     batch_size=20, n_steps=SIM_TIME)  # Initialize a new model structure
    # Load pretrained model weights from path
    meta_saved_model = PPO.load(meta_model_path)
    # Transfer weights to the new model structure
    meta_model.policy.load_state_dict(meta_saved_model.policy.state_dict())
    return meta_model

# Function to generate a variety of demand and lead time scenarios for testing


def make_scenario():
    demand_scenario = []  # List to store demand scenarios
    leadtime_scenario = []  # List to store lead time scenarios
    for mean in range(12, 15):  # Range of mean demand values
        demand_dict = {}
        demand_dict["Dist_Type"] = "GAUSSIAN"  # Demand distribution type
        demand_dict["mean"] = mean  # Mean value of demand
        for std in range(2):  # Range of standard deviations
            demand_dict["std"] = std  # Standard deviation of demand
            demand_scenario.append(demand_dict)  # Append each demand scenario

    for mean in range(2):  # Range of mean lead time values
        leadtime_dict = {}
        leadtime_dict["Dist_Type"] = "GAUSSIAN"  # Lead time distribution type
        leadtime_dict["mean"] = mean  # Mean value of lead time
        for std in range(1):  # Range of standard deviations
            leadtime_dict["std"] = std  # Standard deviation of lead time
            # Append each lead time scenario
            leadtime_scenario.append(leadtime_dict)

    return demand_scenario, leadtime_scenario


# Generate demand and lead time scenarios
demand_scenario, leadtime_scenario = make_scenario()

# Loop over each demand and lead time scenario combination for experiments
case_num = 1
for demand_scenario_dict in demand_scenario:
    for leadtime_scenario_dict in leadtime_scenario:

        # Initialize RL environment and configure scenario settings
        rl_env = GymInterface()
        rl_env.scenario["DEMAND"] = demand_scenario_dict
        rl_env.scenario['LEADTIME'] = leadtime_scenario_dict
        # Path for RL experiment logs
        rl_log_path = os.path.join(EXPERIMENT_LOGS, f'RANDOM_case_{case_num}')
        os.makedirs(rl_log_path)  # Create directory for logs
        # Initialize TensorBoard writer
        rl_env.writer = SummaryWriter(rl_log_path)

        # Build and train RL model with evaluation callback
        rl_model = build_model(rl_env)
        rl_callback = make_call_back(rl_env)
        rl_model.learn(total_timesteps=SIM_TIME * N_EPISODES,
                       callback=rl_callback)  # Train model

        # Initialize Meta environment and configure scenario settings
        meta_env = GymInterface()
        meta_env.scenario["DEMAND"] = demand_scenario_dict
        meta_env.scenario['LEADTIME'] = leadtime_scenario_dict
        # Path for Meta experiment logs
        meta_log_path = os.path.join(EXPERIMENT_LOGS, f'META_case_{case_num}')
        os.makedirs(meta_log_path)  # Create directory for logs
        # Initialize TensorBoard writer
        meta_env.writer = SummaryWriter(meta_log_path)

        # Load pre-trained meta model and set evaluation callback for training
        meta_model = load_model(meta_env)
        meta_callback = make_call_back(meta_env)
        meta_model.learn(total_timesteps=SIM_TIME * N_EPISODES,
                         callback=meta_callback)  # Train model with adaptation

        # Store evaluation rewards from both RL and Meta models for analysis
        experiment_result[f'Case {case_num:02}'] = []
        for x in range(len(rl_callback.rewards)):
            for y in range(2):
                experiment_result[f'Case {case_num:02}'].append(
                    rl_callback.rewards[x])  # Store RL model rewards
                experiment_result[f'Case {case_num:02}'].append(
                    meta_callback.rewards[x])  # Store Meta model rewards
        case_num += 1  # Increment case number for each scenario combination

    # Convert experiment results to a DataFrame and save as a CSV file
    # Transpose results for column labeling
    df = pd.DataFrame(experiment_result).T
    df.columns = ['RL_1000', 'META_1000', 'RL_2000', 'META_2000', 'RL_3000', 'META_3000',
                  'RL_4000', 'META_4000', 'RL_5000', 'META_5000']  # Columns for each evaluation step
    # Save DataFrame as CSV file for analysis
    df.to_csv(os.path.join(RESULT_CSV_EXPERIMENT, 'experiment_result.csv'))
