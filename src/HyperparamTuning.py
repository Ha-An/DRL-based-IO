from config_SimPy import *
from config_RL import *
import optuna
import optuna.visualization as vis
if RL_ALGORITHM == "DQN":
    from stable_baselines3 import DQN
elif RL_ALGORITHM == "DDPG":
    from stable_baselines3 import DDPG
elif RL_ALGORITHM == "PPO":
    from stable_baselines3 import PPO
from InventoryMgtEnv import GymInterface


def tuning_hyperparam(env, trial):
    # Initialize the environment
    env.reset()
    # Define search space for hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
    batch_size = trial.suggest_categorical(
        'batch_size', [16, 32, 64, 128, 256])
    # Define the RL model
    if RL_ALGORITHM == "DQN":
        model = DQN("MlpPolicy", env, learning_rate=learning_rate,
                    gamma=gamma, batch_size=batch_size, verbose=0)
    elif RL_ALGORITHM == "DDPG":
        model = DDPG("MlpPolicy", env, learning_rate=learning_rate,
                     gamma=gamma, batch_size=batch_size, verbose=0)
    elif RL_ALGORITHM == "PPO":
        model = PPO("MlpPolicy", env, learning_rate=learning_rate,
                    gamma=gamma, batch_size=batch_size, verbose=0)
    # Train the model
    model.learn(total_timesteps=SIM_TIME*N_EPISODES)
    # Evaluate the model
    eval_env = GymInterface()
    mean_reward, _ = evaluate_policy(
        model, eval_env, n_eval_episodes=N_EVAL_EPISODES)

    return -mean_reward  # Minimize the negative of mean reward


def run_optuna(env):
    study = optuna.create_study()
    study.optimize(env, tuning_hyperparam, n_trials=N_TRIALS)

    # Print the result
    best_params = study.best_params
    print("Best hyperparameters:", study.best_params)
    # Visualize hyperparameter optimization process
    vis.plot_optimization_history(study).show()
    vis.plot_parallel_coordinate(study).show()
    vis.plot_slice(study).show()
    vis.plot_contour(study, params=['learning_rate', 'gamma']).show()
