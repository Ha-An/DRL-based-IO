import os
from config_SimPy import *

# RL algorithms
RL_ALGORITHM = "PPO"  # "DP", "DQN", "DDPG", "PPO", "SAC"
ACTION_SPACE = [0, 1, 2, 3, 4, 5]

# State space
STATE_DEMAND = True  # True: Demand quantity is included in the state space
# Find minimum Delta
DELTA_MIN = 0
for key in P:
    DELTA_MIN = max(P[key]["PRODUCTION_RATE"] *
                    max(P[key]['QNTY_FOR_INPUT_ITEM']), DEMAND_QTY_MAX)
# maximum production
EXPECTED_PRODUCT_MAX = I[0]['CUST_ORDER_CYCLE']*P[0]['PRODUCTION_RATE']
# Episode
N_EPISODES = 1  # 3000


def DEFINE_FOLDER(folder_name):
    if os.path.exists(folder_name):
        file_list = os.listdir(folder_name)
        folder_name = os.path.join(folder_name, f"Train_{len(file_list)+1}")
    else:
        folder_name = os.path.join(folder_name, "Train_1")

    return folder_name


BEST_PARAMS = {'learning_rate': 0.00012381151768747168,
               'gamma':  0.01, 'batch_size': 256}

# Hyperparameter optimization
OPTIMIZE_HYPERPARAMETERS = False
N_TRIALS = 50  # 50

# Evaluation
N_EVAL_EPISODES = 10  # 100

# Export files
DAILY_REPORT_EXPORT = False
XAI_TRAIN_EXPORT = False

# Define parent dir's path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
# Define each dir's parent dir's path
tensorboard_folder = os.path.join(parent_dir, "tensorboard_log")
result_csv_folder = os.path.join(parent_dir, "result_CSV")
XAI_folder = os.path.join(result_csv_folder, "XAI_Train")
daily_report_folder = os.path.join(result_csv_folder, "daily_report")
graph_folder = os.path.join(result_csv_folder, "Graph")
# Define dir's path
TENSORFLOW_LOGS = DEFINE_FOLDER(tensorboard_folder)
XAI_TRAIN = DEFINE_FOLDER(XAI_folder)
REPORT_LOGS = DEFINE_FOLDER(daily_report_folder)
GRAPH_FOLDER = DEFINE_FOLDER(graph_folder)
# Makedir
if os.path.exists(XAI_TRAIN):
    pass
else:
    os.makedirs(XAI_TRAIN)

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


#XAI
FEATUER_NAME = ["Product", "R1" , "Demamd", "expected shortage"]