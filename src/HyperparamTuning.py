import GymWrapper as gw
import optuna.visualization as vis
import optuna
from config_SimPy import *
from config_RL import *
from stable_baselines3 import DQN, DDPG, PPO
from GymWrapper import GymInterface


def tuning_hyperparam(trial):
    # Initialize the environment
    env = GymInterface()
    env.reset()
    # Define search space for hyperparameters
    LEARNING_RATE = trial.suggest_loguniform('learning_rate', 1e-5, 1)
    GAMMA = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
    BATCH_SIZE = trial.suggest_categorical(
        'batch_size', [16, 32, 64])
    N_STEPS = trial.suggest_categorical(
        'n_steps', [SIM_TIME, SIM_TIME*2, SIM_TIME*3, SIM_TIME*4])
    # Define the RL model

    if RL_ALGORITHM == "PPO":
        model = PPO("MlpPolicy", env, learning_rate=LEARNING_RATE,
                    gamma=GAMMA, batch_size=BATCH_SIZE, n_steps=N_STEPS, verbose=0)
    elif RL_ALGORITHM == "DQN":
        pass
    elif RL_ALGORITHM == "DDPG":
        pass

    # Train the model
    model.learn(total_timesteps=SIM_TIME*N_EPISODES)
    # Evaluate the model
    eval_env = GymInterface()
    mean_reward, _ = gw.evaluate_model(model, eval_env, N_EVAL_EPISODES)

    return -mean_reward  # Minimize the negative of mean reward


def run_optuna():
    # study = optuna.create_study( )
    study = optuna.create_study(direction='minimize')
    study.optimize(tuning_hyperparam, n_trials=N_TRIALS)

    # Print the result
    best_params = study.best_params
    print("Best hyperparameters:", best_params)
    # Visualize hyperparameter optimization process
    vis.plot_optimization_history(study).show()
    vis.plot_parallel_coordinate(study).show()
    vis.plot_slice(study).show()
    vis.plot_contour(study, params=['learning_rate', 'gamma']).show()
