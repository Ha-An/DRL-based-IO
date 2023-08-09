# Items:
# ID: Index of the element in the dictionary
# TYPE: Product, Raw Material, WIP;
# NAME: Item's name or model;
# INIT_LEVEL: Initial inventory level [units] 
# CUST_ORDER_CYCLE: Customer ordering cycle [days]
# MANU_ORDER_CYCLE: Manufacturer ordering cycle to providers [days]
# DEMAND_QUANTITY: Demand quantity for the final product [units]
# DELIVERY_TIME_TO_CUST: Delivery time to the customer [days]
# DELIVERY_TIME_FROM_SUP: Delivery time from a supplier [days]
# MANU_LEAD_TIME: The total processing time for the manufacturer to process and deliver the customer's order [days]
# SUP_LEAD_TIME: The total processing time for a supplier to process and deliver the manufacturer's order [days]
# LOT_SIZE_ORDER: Lot-size for the order of raw materials (Q) [units]
# HOLD_COST: Holding cost of the items [$/unit*day]
# SHORTAGE_COST: Shortage cost of items [$/unit]
# PURCHASE_COST: Holding cost of the raw materials [$/unit]
# SETUP_COST_PRO: Setup cost for the delivery of the products to the customer [$/delivery]
# SETUP_COST_RAW: Setup cost for the ordering of the raw materials to a supplier [$/order]
# DELIVERY_COST: Delivery cost of the products [$/unit]

I = {0: {"ID": 0, "TYPE": "Product",      "NAME": "PRODUCT",          "INIT_LEVEL": 30, "CUST_ORDER_CYCLE": 7, "DEMAND_QUANTITY": 21, "MANU_LEAD_TIME": 7,                      "HOLD_COST": 5, "SHORTAGE_COST": 10,                     "SETUP_COST_PRO": 50, "DELIVERY_COST": 10},
     1: {"ID": 1, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 1.1", "INIT_LEVEL": 30, "MANU_ORDER_CYCLE": 7,                        "SUP_LEAD_TIME": 7, "LOT_SIZE_ORDER": 21, "HOLD_COST": 1, "SHORTAGE_COST": 2, "PURCHASE_COST": 3,  "SETUP_COST_RAW": 20},
     2: {"ID": 2, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 2.1", "INIT_LEVEL": 30, "MANU_ORDER_CYCLE": 7,                        "SUP_LEAD_TIME": 7, "LOT_SIZE_ORDER": 21, "HOLD_COST": 1, "SHORTAGE_COST": 2, "PURCHASE_COST": 3,  "SETUP_COST_RAW": 20},
     3: {"ID": 3, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 2.2", "INIT_LEVEL": 30, "MANU_ORDER_CYCLE": 7,                        "SUP_LEAD_TIME": 7, "LOT_SIZE_ORDER": 21, "HOLD_COST": 1, "SHORTAGE_COST": 2, "PURCHASE_COST": 3,  "SETUP_COST_RAW": 20},
     4: {"ID": 4, "TYPE": "WIP",          "NAME": "WIP 1",            "INIT_LEVEL": 30,                                                                                         "HOLD_COST": 2, "SHORTAGE_COST": 2}}

# Processes:
# ID: Index of the element in the dictionary
# PRODUCTION_RATE [units/day]
# INPUT_LIST: List of input materials or WIPs
# OUTPUT: Output WIP or Product
# PROCESS_COST: Processing cost of the process [$/day]
P = {0: {"ID": 0, "PRODUCTION_RATE": 3, "INPUT_LIST": [I[1]],             "OUTPUT": I[4], "PROCESS_COST": 5},
     1: {"ID": 1, "PRODUCTION_RATE": 2, "INPUT_LIST": [I[2], I[3], I[4]], "OUTPUT": I[0], "PROCESS_COST": 6}}


Ver_print = False
# Simulation
SIM_TIME = 30# [days]
INITIAL_INVENTORY = 30 # [units]
total_cost_per_day = []
EPISODES = 2
batch_size = 32
action_space = []
values = [0, 20, 40]
for i in values:
    for j in values:
        for k in values:
            action_space.append([i, j, k])
