import simpy
import random
from visualization import *
from DQN import *
# Items:
# ID: Index of the element in the dictionary
# TYPE: Product, Raw Material, WIP;
# NAME: Item's name or model;
# REMOVE## FROM: process key; TO: process key;
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

I = {0: {"ID": 0, "TYPE": "Product",      "NAME": "PRODUCT",          "CUST_ORDER_CYCLE": 1, "DEMAND_QUANTITY": 21, "MANU_LEAD_TIME": 3,                      "HOLD_COST": 5, "SHORTAGE_COST": 10,                     "SETUP_COST_PRO": 50, "DELIVERY_COST": 10},
     1: {"ID": 1, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 1.1", "MANU_ORDER_CYCLE": 1,                        "SUP_LEAD_TIME": 3, "LOT_SIZE_ORDER": 21, "HOLD_COST": 1, "SHORTAGE_COST": 2, "PURCHASE_COST": 3,  "SETUP_COST_RAW": 20},
     2: {"ID": 2, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 2.1", "MANU_ORDER_CYCLE": 1,                        "SUP_LEAD_TIME": 3, "LOT_SIZE_ORDER": 21, "HOLD_COST": 1, "SHORTAGE_COST": 2, "PURCHASE_COST": 3,  "SETUP_COST_RAW": 20},
     3: {"ID": 3, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 2.2", "MANU_ORDER_CYCLE": 1,                        "SUP_LEAD_TIME": 3, "LOT_SIZE_ORDER": 21, "HOLD_COST": 1, "SHORTAGE_COST": 2, "PURCHASE_COST": 3,  "SETUP_COST_RAW": 20},
     4: {"ID": 4, "TYPE": "WIP",          "NAME": "WIP 1",                                                                                                    "HOLD_COST": 2, "SHORTAGE_COST": 2}}

# Processes:
# ID: Index of the element in the dictionary
# PRODUCTION_RATE [units/day]
# INPUT_LIST: List of input materials or WIPs
# OUTPUT: Output WIP or Product
# PROCESS_COST: Processing cost of the process [$/day]
P = {0: {"ID": 0, "PRODUCTION_RATE": 3, "INPUT_LIST": [I[1]],             "OUTPUT": I[4], "PROCESS_COST": 5},
     1: {"ID": 1, "PRODUCTION_RATE": 2, "INPUT_LIST": [I[2], I[3], I[4]], "OUTPUT": I[0], "PROCESS_COST": 6}}

# Demand quantity for the final product [units]
'''
MIN_ORDER_SIZE = 80
MAX_ORDER_SIZE = 120
'''
Ver_print = False
# Simulation
SIM_TIME = 100  # [days]
INITIAL_INVENTORY = 30  # [units]
total_cost_per_day = []
EPISODES = 1000
batch_size = 32
action_space = []
values = [0, 20, 40]
for i in values:
    for j in values:
        for k in values:
            action_space.append([i, j, k])


class Inventory:
    def __init__(self, env, item_id, holding_cost, shortage_cost):
        self.item_id = item_id  # 0: product; others: WIP or raw material
        self.level = INITIAL_INVENTORY  # capacity=infinity
        self.holding_cost = holding_cost  # $/unit*day
        self.shortage_cost = shortage_cost
        self.level_over_time = []  # Data tracking for inventory level
        self.inventory_cost_over_time = []  # Data tracking for inventory cost

    def cal_inventory_cost(self):
        if self.level > 0:
            self.inventory_cost_over_time.append(
                self.holding_cost * self.level)
        elif self.level < 0:
            self.inventory_cost_over_time.append(
                self.shortage_cost * abs(self.level))
        else:
            self.inventory_cost_over_time.append(0)
        if Ver_print:
            print(
                f"[Inventory Cost of {I[self.item_id]['NAME']}]  {self.inventory_cost_over_time[-1]}")


class Provider:
    def __init__(self, env, name, item_id):
        self.env = env
        self.name = name
        self.item_id = item_id

    def deliver(self, order_size, inventory):
        # Lead time
        yield self.env.timeout(I[self.item_id]["SUP_LEAD_TIME"] * 24)
        inventory.level += order_size
        if Ver_print:
            print(
                f"{self.env.now}: {self.name} has delivered {order_size} units of {I[self.item_id]['NAME']}")


class Procurement:
    def __init__(self, env, item_id, purchase_cost, setup_cost):
        self.env = env
        self.item_id = item_id
        self.purchase_cost = purchase_cost
        self.setup_cost = setup_cost
        self.purchase_cost_over_time = []  # Data tracking for purchase cost
        self.setup_cost_over_time = []  # Data tracking for setup cost
        self.daily_procurement_cost = 0

    def order(self, provider, inventory):
        while True:
            # Place an order to a provider
            yield self.env.timeout(I[self.item_id]["MANU_ORDER_CYCLE"] * 24)
            # THIS WILL BE AN ACTION OF THE AGENT
            order_size = I[self.item_id]["LOT_SIZE_ORDER"]
            if Ver_print:
                print(
                    f"{self.env.now}: Placed an order for {order_size} units of {I[self.item_id]['NAME']}")
            self.env.process(provider.deliver(order_size, inventory))
            self.cal_procurement_cost()

    def cal_procurement_cost(self):
        self.daily_procurement_cost += self.purchase_cost * \
            I[self.item_id]["LOT_SIZE_ORDER"] + self.setup_cost

    def cal_daily_procurement_cost(self):
        if Ver_print:
            print(
                f"[Daily procurement cost of {I[self.item_id]['NAME']}]  {self.daily_procurement_cost}")
        self.daily_procurement_cost = 0


class Production:
    def __init__(self, env, name, process_id, production_rate, output, input_inventories, output_inventory, processing_cost):
        self.env = env
        self.name = name
        self.process_id = process_id
        self.production_rate = production_rate
        self.output = output
        self.input_inventories = input_inventories
        self.output_inventory = output_inventory
        self.processing_cost = processing_cost
        self.processing_cost_over_time = []  # Data tracking for processing cost
        self.daily_production_cost = 0

    def process(self):
        while True:
            # Check the current state if input materials or WIPs are available
            shortage_check = False
            for inven in self.input_inventories:
                if inven.level < 1:
                    inven.level -= 1
                    shortage_check = True
            if shortage_check:
                if Ver_print:
                    print(
                        f"{self.env.now}: Stop {self.name} due to a shortage of input materials or WIPs")
                # Check again after 24 hours (1 day)
                yield self.env.timeout(24)
                # continue
            else:
                # Consuming input materials or WIPs and producing output WIP or Product
                processing_time = 24 / self.production_rate
                yield self.env.timeout(processing_time)
                if Ver_print:
                    print(f"{self.env.now}: Process {self.process_id} begins")
                for inven in self.input_inventories:
                    inven.level -= 1
                    if Ver_print:
                        print(
                            f"{self.env.now}: Inventory level of {I[inven.item_id]['NAME']}: {inven.level}")
                self.output_inventory.level += 1
                self.cal_processing_cost(processing_time)
                if Ver_print:
                    print(
                        f"{self.env.now}: A unit of {self.output['NAME']} has been produced")
                    print(
                        f"{self.env.now}: Inventory level of {I[self.output_inventory.item_id]['NAME']}: {self.output_inventory.level}")

    def cal_processing_cost(self, processing_time):
        self.daily_production_cost += self.processing_cost * processing_time

    def cal_daily_production_cost(self):
        if Ver_print:
            print(
                f"[Daily production cost of {self.name}]  {self.daily_production_cost}")
        self.daily_production_cost = 0


class Sales:
    def __init__(self, env, item_id, delivery_cost, setup_cost):
        self.env = env
        self.item_id = item_id
        self.delivery_cost = delivery_cost
        self.setup_cost = setup_cost
        self.selling_cost_over_time = []  # Data tracking for selling cost
        self.daily_selling_cost = 0

    def delivery(self, item_id, order_size, product_inventory):
        # Lead time
        yield self.env.timeout(I[item_id]["MANU_LEAD_TIME"] * 24)
        # SHORTAGE: Check if products are available
        if product_inventory.level < order_size:
            num_shortages = abs(product_inventory.level - order_size)
            if product_inventory.level > 0:
                if Ver_print:
                    print(
                        f"{self.env.now}: {product_inventory.level} units of the product have been delivered to the customer")
                # yield self.env.timeout(DELIVERY_TIME)
                product_inventory.level -= order_size
                self.cal_selling_cost()
            if Ver_print:
                print(
                    f"{self.env.now}: Unable to deliver {num_shortages} units to the customer due to product shortage")
            # Check again after 24 hours (1 day)
            # yield self.env.timeout(24)
        # Delivering products to the customer
        else:
            product_inventory.level -= order_size
            if Ver_print:
                print(
                    f"{self.env.now}: {order_size} units of the product have been delivered to the customer")
            self.cal_selling_cost()

    def cal_selling_cost(self):
        self.daily_selling_cost += self.delivery_cost * \
            I[self.item_id]['DEMAND_QUANTITY'] + self.setup_cost

    def cal_daily_selling_cost(self):
        if Ver_print:
            print(
                f"[Daily selling cost of  {I[self.item_id]['NAME']}]  {self.daily_selling_cost}")
            self.daily_selling_cost = 0


class Customer:
    def __init__(self, env, name, item_id):
        self.env = env
        self.name = name
        self.item_id = item_id
        self.order_history = []

    def order(self, sales, product_inventory):
        while True:
            yield self.env.timeout(I[self.item_id]["CUST_ORDER_CYCLE"] * 24)
            # THIS WILL BE A RANDOM VARIABLE
            order_size = I[self.item_id]["DEMAND_QUANTITY"]
            self.order_history.append(order_size)
            if Ver_print:
                print(
                    f"{self.env.now}: The customer has placed an order for {order_size} units of {I[self.item_id]['NAME']}")
            self.env.process(sales.delivery(
                self.item_id, order_size, product_inventory))
    ''' 
    def delivery(self, product_inventory):
        while True:
            # SHORTAGE: Check products are available
            if len(product_inventory.store.items) < 1:
                print(
                    f"{self.env.now}: Unable to deliver to the customer due to product shortage")
                # Check again after 24 hours (1 day)
                yield self.env.timeout(24)
            # Delivering products to the customer
            else:
                demand = I[product_inventory.item_id]["DEMAND_QUANTITY"]
                for _ in range(demand):
                    yield product_inventory.store.get()
                print(
                    f"{self.env.now}: {demand} units of the product have been delivered to the customer")
    '''


'''
def calculate_inventory_cost():
    for item, quantity in inventory.items():
        if quantity > 0:
            inventory_cost[item] += HOLDING_COST * quantity
        else:
            inventory_cost[item] += BACKORDER_COST * abs(quantity)
        inventory_cost_over_time[item].append(inventory_cost[item])
'''

'''
take action에 대한 코드 
item_ids = [item_id for item_id, item_info in I.items() if 'LOT_SIZE_ORDER' in item_info]

def take_action(actions, item_ids):
    for i, item_id in enumerate(item_ids):
        I[item_id]["LOT_SIZE_ORDER"] = actions[i]
'''


def take_action(action_space, action, env, inventoryList, total_cost_per_day, I):
    seq = -1
    for items in range(len(I)):
        if 'LOT_SIZE_ORDER' in I[items]:
            seq += 1
            if type(action) != list:
                for a in range(len(action_space)):
                    if action_space[action] == action_space[a]:
                        order_size = action_space[action]
                        I[items]['LOT_SIZE_ORDER'] = order_size[seq]
            else:
                order_size = action
                I[items]['LOT_SIZE_ORDER'] = order_size[seq]

        # print(
        #     f"{env.now}: Placed an order for {order_size[seq]} units of {I[items.item_id]['NAME']}")
    # Run the simulation for one day (24 hours)
    env.run(until=env.now + 24)

    # Calculate the next state after the actions are taken
    next_state = np.array([inven.level for inven in inventoryList])

    # Calculate the reward and whether the simulation is done
    # You need to define this function based on your specific reward policy
    reward = -total_cost_per_day[-1]
    # Terminate the episode if the simulation time is reached
    done = (env.now >= SIM_TIME * 24)

    return next_state, reward, done


def main():

    env = simpy.Environment()
    # Print the list of items and processes
    print("\nItem list")
    for i in I.keys():
        print(f"ITEM {i}: {I[i]['NAME']}")
    print("\nProcess list")
    for i in P.keys():
        print(f"Output of PROCESS {i}: {P[i]['OUTPUT']['NAME']}")

    # Create an inventory for each item
    inventoryList = []
    for i in I.keys():
        inventoryList.append(
            Inventory(env, i, I[i]["HOLD_COST"], I[i]["SHORTAGE_COST"]))
    print("Number of Inventories: ", len(inventoryList))

    # Create stakeholders (Customer, Providers)
    customer = Customer(env, "CUSTOMER", I[0]["ID"])
    providerList = []
    procurementList = []
    for i in I.keys():
        # Create a provider and the corresponding procurement if the type of the item is Raw Material
        if I[i]["TYPE"] == 'Raw Material':
            providerList.append(Provider(env, "PROVIDER_"+str(i), i))
            procurementList.append(Procurement(
                env, I[i]["ID"], I[i]["PURCHASE_COST"], I[i]["SETUP_COST_RAW"]))
    print("Number of Providers: ", len(providerList))

    # Create managers for manufacturing process, procurement process, and delivery process
    sales = Sales(env, customer.item_id,
                  I[0]["DELIVERY_COST"], I[0]["SETUP_COST_PRO"])
    productionList = []
    for i in P.keys():
        output_inventory = inventoryList[P[i]["OUTPUT"]["ID"]]
        input_inventories = []
        for j in P[i]["INPUT_LIST"]:
            input_inventories.append(inventoryList[j["ID"]])
        productionList.append(Production(env, "PROCESS_"+str(i), P[i]["ID"],
                                         P[i]["PRODUCTION_RATE"], P[i]["OUTPUT"], input_inventories, output_inventory, P[i]["PROCESS_COST"]))

    # Event processes for SimPy simulation
    env.process(customer.order(sales, inventoryList[I[0]["ID"]]))
    for production in productionList:
        env.process(production.process())
    for i in range(len(providerList)):
        env.process(procurementList[i].order(
            providerList[i], inventoryList[providerList[i].item_id]))
# 코드에 들어가는 옵션값
    total_cost_per_day = []
    # Run the simulation
    state = np.array([inven.level for inven in inventoryList]
                     )  # Get the inventory levels
    state_size = len(inventoryList)  # Number of inventories
    agent = DQNAgent(state_size, action_space)
    done = (env.now >= SIM_TIME * 24)
    total_rewards, losses = [], []
    total_reward = 0
    for episode in range(EPISODES):
        for i in range(SIM_TIME*24+24):
            # Print the inventory level every 24 hours (1 day)
            if i % 24 == 0:
                if i != 0:
                    if Ver_print:
                        print("day", i/24)
                    # Calculate the cost models
                    for inven in inventoryList:
                        inven.cal_inventory_cost()
                    for production in productionList:
                        production.cal_daily_production_cost()
                    for procurement in procurementList:
                        procurement.cal_daily_procurement_cost()
                    sales.cal_daily_selling_cost()
                    # Calculate the total cost for the current day and append to the list
                    total_cost = 0
                    for inven in inventoryList:
                        total_cost += sum(inven.inventory_cost_over_time)
                    for production in productionList:
                        total_cost += production.daily_production_cost
                    for procurement in procurementList:
                        total_cost += procurement.daily_procurement_cost
                    total_cost += sales.daily_selling_cost
                    total_cost_per_day.append(total_cost)

                    # 하루단위로 보상을 계속 받아서 업데이트 해주기 위해서 리셋을 진행하는 코드
                    for inven in inventoryList:
                        inven.inventory_cost_over_time = []
                    for production in productionList:
                        production.daily_production_cost = 0
                    for procurement in procurementList:
                        procurement.daily_procurement_cost = 0
                    sales.daily_selling_cost = 0
                    action = agent.choose_action(state)
                    next_state, reward, done = take_action(
                        action_space, action, env, inventoryList, total_cost_per_day, I)
                    agent.remember(Transition(
                        state, action, reward, next_state, done))
                    if Ver_print:
                        print("done :", done)

                    state = next_state

                    if len(agent.memory) == agent.max_memory_size:
                        loss = agent.replay(batch_size)
                        losses.append(loss)

                    total_reward += reward

                    if done:
                        if Ver_print:
                            print(
                                "_________________________________________________________done")
                        total_rewards.append(total_reward)
                        print(
                            f'Episode: {episode}/{EPISODES}, Total Reward: {total_reward}')
                        total_reward = 0  # 리워드 초기화
                        env = simpy.Environment()  # 환경초기화
                        # Create an inventory for each item
                        inventoryList = []
                        for i in I.keys():
                            inventoryList.append(
                                Inventory(env, i, I[i]["HOLD_COST"], I[i]["SHORTAGE_COST"]))

                        # Create stakeholders (Customer, Providers)
                        customer = Customer(env, "CUSTOMER", I[0]["ID"])
                        providerList = []
                        procurementList = []
                        for i in I.keys():
                            # Create a provider and the corresponding procurement if the type of the item is Raw Material
                            if I[i]["TYPE"] == 'Raw Material':
                                providerList.append(
                                    Provider(env, "PROVIDER_"+str(i), i))
                                procurementList.append(Procurement(
                                    env, I[i]["ID"], I[i]["PURCHASE_COST"], I[i]["SETUP_COST_RAW"]))

                        # Create managers for manufacturing process, procurement process, and delivery process
                        sales = Sales(env, customer.item_id,
                                      I[0]["DELIVERY_COST"], I[0]["SETUP_COST_PRO"])
                        productionList = []
                        for i in P.keys():
                            output_inventory = inventoryList[P[i]
                                                             ["OUTPUT"]["ID"]]
                            input_inventories = []
                            for j in P[i]["INPUT_LIST"]:
                                input_inventories.append(
                                    inventoryList[j["ID"]])
                            productionList.append(Production(env, "PROCESS_"+str(i), P[i]["ID"],
                                                             P[i]["PRODUCTION_RATE"], P[i]["OUTPUT"], input_inventories, output_inventory, P[i]["PROCESS_COST"]))

                        # Event processes for SimPy simulation
                        env.process(customer.order(
                            sales, inventoryList[I[0]["ID"]]))
                        for production in productionList:
                            env.process(production.process())
                        for i in range(len(providerList)):
                            env.process(procurementList[i].order(
                                providerList[i], inventoryList[providerList[i].item_id]))
                        state = np.array([inven.level for inven in inventoryList]
                                         )  # Get the initial inventory levels
                        break
    # Print the inventory level
    '''
                print(f"\nDAY {int(i/24)+1}")
                for inven in inventoryList:
                    inven.level_over_time.append(inven.level)
                    print(
                        f"[{I[inven.item_id]['NAME']}]  {inven.level}")
    '''
    print(total_rewards)
    visualization.plot_learning_history(total_rewards)
    '''
    # Visualize the data trackers of the inventory level and cost over time
    for i in I.keys():
        inventory_visualization = visualization.visualization(
            inventoryList[i], I[i]['NAME'])
        inventory_visualization.inventory_level_graph()
        inventory_visualization.inventory_cost_graph()
        # calculate_inventory_cost()
    '''


if __name__ == "__main__":
    main()
