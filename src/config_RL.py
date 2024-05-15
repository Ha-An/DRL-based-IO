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
DELTA_MIN = 0
for key in P:
    DELTA_MIN = max(P[key]["PRODUCTION_RATE"] *
                    max(P[key]['QNTY_FOR_INPUT_ITEM']), DEMAND_QTY_MAX)
# maximum production
EXPECTED_PRODUCT_MAX = I[0]['CUST_ORDER_CYCLE']*P[0]['PRODUCTION_RATE']

# Episode
N_EPISODES = 1800  # 3000


def DEFINE_FOLDER(folder_name):
    if os.path.exists(folder_name):
        file_list = os.listdir(folder_name)
        folder_name = os.path.join(folder_name, f"Train_{len(file_list)+1}")
    else:
        folder_name = os.path.join(folder_name, "Train_1")

    return folder_name


# BEST_PARAMS = {'learning_rate': 0.00012381151768747168,
#                'gamma':  0.01, 'batch_size': 256}

# Hyperparameter optimization
OPTIMIZE_HYPERPARAMETERS = False
N_TRIALS = 100  # 50

# Evaluation
N_EVAL_EPISODES = 10  # 100

# Export files
DAILY_REPORT_EXPORT = False
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
graph_folder = os.path.join(result_csv_folder, "Graph")

# Define dir's path
TENSORFLOW_LOGS = DEFINE_FOLDER(tensorboard_folder)
STATE = DEFINE_FOLDER(STATE_folder)
REPORT_LOGS = DEFINE_FOLDER(daily_report_folder)
GRAPH_FOLDER = DEFINE_FOLDER(graph_folder)

# Makedir
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
# tensorboard --logdir="~\tensorboard_log"
# http://localhost:6006/
#Validation 
VALIDATION_PRINT=True

