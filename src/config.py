#### Items #####################################################################
# ID: Index of the element in the dictionary
# TYPE: Product, Material, WIP;
# NAME: Item's name or model;
# CUST_ORDER_CYCLE: Customer ordering cycle [days]
# MANU_ORDER_CYCLE: Manufacturer ordering cycle to suppliers [days]
# DEMAND_QUANTITY: Demand quantity for the final product [units] -> THIS IS UPDATED EVERY 24 HOURS (Default: 0)
# DELIVERY_TIME_TO_CUST: Delivery time to the customer [days]
# DELIVERY_TIME_FROM_SUP: Delivery time from a supplier [days]
# SUP_LEAD_TIME: The total processing time for a supplier to process and deliver the manufacturer's order [days]
# REMOVE## LOT_SIZE_ORDER: Lot-size for the order of materials (Q) [units] -> THIS IS AN AGENT ACTION THAT IS UPDATED EVERY 24 HOURS
# HOLD_COST: Holding cost of the items [$/unit*day]
# PURCHASE_COST: Holding cost of the materials [$/unit]
# SETUP_COST_PRO: Setup cost for the delivery of the products to the customer [$/delivery]
# ORDER_COST_TO_SUP: Ordering cost for the materials to a supplier [$/order]
# DELIVERY_COST: Delivery cost of the products [$/unit]
# DUE_DATE: Term of customer order to delivered [days]
# SHORTAGE_COST: Backorder cost of products [$/unit]

#### Processes #####################################################################
# ID: Index of the element in the dictionary
# PRODUCTION_RATE [units/day]
# INPUT_TYPE_LIST: List of types of input materials or WIPs
# QNTY_FOR_INPUT_ITEM: Quantity of input materials or WIPs [units]
# OUTPUT: Output WIP or Product
# PROCESS_COST: Processing cost of the process [$/processtime]
# PROCESS_STOP_COST: Penalty cost for stopping the process [$/unit]


# Scenario 1
'''
I = {0: {"ID": 0, "TYPE": "Product",      "NAME": "PRODUCT",
         "CUST_ORDER_CYCLE": 7,
         "DEMAND_QUANTITY": 0,
         "HOLD_COST": 1,
         "SETUP_COST_PRO": 1,
         "DELIVERY_COST": 1,
         "DUE_DATE": 7,
         "SHORTAGE_COST_PRO": 50},
     1: {"ID": 1, "TYPE": "Material", "NAME": "MATERIAL 1",
         "MANU_ORDER_CYCLE": 1,
         "SUP_LEAD_TIME": 2,  # SUP_LEAD_TIME must be an integer
         "HOLD_COST": 1,
         "PURCHASE_COST": 2,
         "ORDER_COST_TO_SUP": 1,
         "LOT_SIZE_ORDER": 0}}

P = {0: {"ID": 0, "PRODUCTION_RATE": 2, "INPUT_TYPE_LIST": [I[1]], "QNTY_FOR_INPUT_ITEM": [
    1], "OUTPUT": I[0], "PROCESS_COST": 1, "PROCESS_STOP_COST": 2}}


'''
# Scenario 2
I = {0: {"ID": 0, "TYPE": "Product",      "NAME": "PRODUCT",
         "CUST_ORDER_CYCLE": 7,
         "DEMAND_QUANTITY": 0,
         "HOLD_COST": 3,
         "SETUP_COST_PRO": 5,
         "DELIVERY_COST": 1,
         "DUE_DATE": 5,
         "SHORTAGE_COST_PRO": 2},
     1: {"ID": 1, "TYPE": "Material", "NAME": "MATERIAL 1.1",
         "MANU_ORDER_CYCLE": 2,
         "SUP_LEAD_TIME": 0,
         "HOLD_COST": 3,
         "PURCHASE_COST": 2,
         "ORDER_COST_TO_SUP": 1},
     2: {"ID": 2, "TYPE": "Material", "NAME": "MATERIAL 2.1",
         "MANU_ORDER_CYCLE": 3,
         "SUP_LEAD_TIME": 0,
         "HOLD_COST": 3,
         "PURCHASE_COST": 2,
         "ORDER_COST_TO_SUP": 1},
     3: {"ID": 3, "TYPE": "Material", "NAME": "MATERIAL 2.2",
         "MANU_ORDER_CYCLE": 4,
         "SUP_LEAD_TIME": 0,
         "HOLD_COST": 3,
         "PURCHASE_COST": 2,
         "ORDER_COST_TO_SUP": 1},
     4: {"ID": 4, "TYPE": "WIP",          "NAME": "WIP 1",
         "HOLD_COST": 1, }}

P = {0: {"ID": 0, "PRODUCTION_RATE": 2,
         "INPUT_TYPE_LIST": [I[1]], "QNTY_FOR_INPUT_ITEM": [1],
         "OUTPUT": I[4],
         "PROCESS_COST": 2,
         "PROCESS_STOP_COST": 2},
     1: {"ID": 1, "PRODUCTION_RATE": 2,
         "INPUT_TYPE_LIST": [I[2], I[3], I[4]], "QNTY_FOR_INPUT_ITEM": [1, 1, 1],
         "OUTPUT": I[0],
         "PROCESS_COST": 2,
         "PROCESS_STOP_COST": 3}}

# RL algorithms
RL_ALGORITHM = "PPO"  # "DP", "DQN", "DDPG", "PPO", "SAC"
ACTION_SPACE = [0, 1, 2, 3, 4, 5]

# State space
RL_ALGORITHM = "PPO"  # "DQN", "DDPG", "PPO"
ACTION_SPACE = [0, 1, 2, 3, 4, 5]
# if this is not 0, the length of state space of demand quantity is not identical to INVEN_LEVEL_MAX
INVEN_LEVEL_MIN = 0
INVEN_LEVEL_MAX = 50  # Capacity limit of the inventory [units]
STATE_DEMAND = True  # True: Demand quantity is included in the state space

# Simulation
N_EPISODES = 100  # 3000
SIM_TIME = 50  # 200 [days] per episode
INIT_LEVEL = 10  # Initial inventory level [units]

# Uncertainty factors
DEMAND_QTY_MIN = 5  # if this is not 0, the length of state space of demand quantity is not identical to DEMAND_QTY_MAX
DEMAND_QTY_MAX = 10
# DUE_DATE_MIN = 0  # if this is not 0, the length of state space of demand quantity is not identical to DUE_DATE_MAX
# DUE_DATE_MAX = 3

# Ordering rules
ORDER_QTY = 15
REORDER_LEVEL = 10

BEST_PARAMS = {'learning_rate': 0.00012381151768747168,
               'gamma':  0.01, 'batch_size': 256}

# Hyperparameter optimization #하이퍼 파라미터 최적화 선택
OPTIMIZE_HYPERPARAMETERS = False
N_TRIALS = 50  # 50
N_EVAL_EPISODES = 100  # 100

# Print logs
PRINT_SIM_EVENTS = True

#Tensorboard
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
TENSORFLOW_LOGS = os.path.join(parent_dir,"tensorboard_log")

# Cost model
# If False, the total cost is calculated based on the inventory level for every 24 hours.
# Otherwise, the total cost is accumulated every hour.
HOURLY_COST_MODEL = True
