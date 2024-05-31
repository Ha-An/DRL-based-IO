import os
from config_SimPy import *

# RL algorithms
RL_ALGORITHM = "PPO"  # "DP", "DQN", "DDPG", "PPO", "SAC"
ACTION_SPACE = [0, 1, 2, 3, 4, 5]

'''
# State space
STATE_RANGES = []
for i in range(len(I)):
    # Inventory level
    STATE_RANGES.append((0, INVEN_LEVEL_MAX))
    # Daily change for the on-hand inventory
    STATE_RANGES.append((-INVEN_LEVEL_MAX, INVEN_LEVEL_MAX))
# Remaining demand: Demand quantity - Current product level
STATE_RANGES.append((0, max(DEMAND_QTY_MAX, INVEN_LEVEL_MAX)))
'''
# Find minimum Delta
PRODUCT_OUTGOING_CORRECTION = 0
for key in P:
    PRODUCT_OUTGOING_CORRECTION = max(P[key]["PRODUCTION_RATE"] *
                    max(P[key]['QNTY_FOR_INPUT_ITEM']), DEMAND_QTY_MAX)
# maximum production

# Episode
N_EPISODES = 4000 # 3000


def DEFINE_FOLDER(folder_name):
    if os.path.exists(folder_name):
        file_list = os.listdir(folder_name)
        folder_name = os.path.join(folder_name, f"Train_{len(file_list)+1}")
    else:
        folder_name = os.path.join(folder_name, "Train_1")

    return folder_name


def save_path(path):
    import shutil

    if os.path.exists(path):
        shutil.rmtree(path)

    # Create a new folder
    os.makedirs(path)
    return path

# BEST_PARAMS = {'learning_rate': 0.00012381151768747168,
#                'gamma':  0.01, 'batch_size': 256}


# Hyperparameter optimization
OPTIMIZE_HYPERPARAMETERS = False
N_TRIALS = 100  # 50

# Evaluation
N_EVAL_EPISODES = 1  # 100

# Export files
DAILY_REPORT_EXPORT = True
STATE_TRAIN_EXPORT = True
STATE_TEST_EXPORT = True

# Define parent dir's path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
# Define each dir's parent dir's path
tensorboard_folder = os.path.join(parent_dir, "tensorboard_log")
result_csv_folder = os.path.join(parent_dir, "result_CSV")
STATE_folder = os.path.join(result_csv_folder, "state")
daily_report_folder = os.path.join(result_csv_folder, "daily_report")

# Define dir's path
TENSORFLOW_LOGS = DEFINE_FOLDER(tensorboard_folder)
'''
STATE = DEFINE_FOLDER(STATE_folder)
REPORT_LOGS = DEFINE_FOLDER(daily_report_folder)
GRAPH_FOLDER = DEFINE_FOLDER(graph_folder)
'''
STATE = save_path(STATE_folder)
REPORT_LOGS = save_path(daily_report_folder)

# Makedir
'''
if os.path.exists(STATE):
    pass
else:
    os.makedirs(STATE)

if os.path.exists(REPORT_LOGS):
    pass
else:
    os.makedirs(REPORT_LOGS)
if os.path.exists(GRAPH_FOLDER):
    pass
else:
    os.makedirs(GRAPH_FOLDER)
'''
#Visualize_Graph
VIZ_INVEN_LINE=True
VIZ_INVEN_PIE=True
VIZ_COST_PIE=True
VIZ_COST_BOX=True

#Saved Model
SAVED_MODEL_PATH=os.path.join(parent_dir,"Saved_Model")
SAVE_MODEL=True
SAVED_MODEL_NAME="PPO_MODEL"

#Load Model
LOAD_MODEL=False
LOAD_MODEL_NAME="PPO_MODEL" 


# Non-stationary demand
mean_demand = 100
standard_deviation_demand = 20


# tensorboard --logdir="~\tensorboard_log"
# http://localhost:6006/
