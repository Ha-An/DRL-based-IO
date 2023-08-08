import simpy
import numpy as np

import random
import visualization
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

I = {0: {"ID": 0, "TYPE": "Product",      "NAME": "PRODUCT",          "CUST_ORDER_CYCLE": 7, "DEMAND_QUANTITY": 21, "MANU_LEAD_TIME": 7,                      "HOLD_COST": 5, "SHORTAGE_COST": 10,                     "SETUP_COST_PRO": 50, "DELIVERY_COST": 10},
     1: {"ID": 1, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 1.1", "MANU_ORDER_CYCLE": 7,                        "SUP_LEAD_TIME": 7, "LOT_SIZE_ORDER": 21, "HOLD_COST": 1, "SHORTAGE_COST": 2, "PURCHASE_COST": 3,  "SETUP_COST_RAW": 20},
     2: {"ID": 2, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 2.1", "MANU_ORDER_CYCLE": 7,                        "SUP_LEAD_TIME": 7, "LOT_SIZE_ORDER": 21, "HOLD_COST": 1, "SHORTAGE_COST": 2, "PURCHASE_COST": 3,  "SETUP_COST_RAW": 20},
     3: {"ID": 3, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 2.2", "MANU_ORDER_CYCLE": 7,                        "SUP_LEAD_TIME": 7, "LOT_SIZE_ORDER": 21, "HOLD_COST": 1, "SHORTAGE_COST": 2, "PURCHASE_COST": 3,  "SETUP_COST_RAW": 20},
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
# Simulation
SIM_TIME = 5  # [days]
INITIAL_INVENTORY = 30  # [units]


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
            print(
                f"{self.env.now}: Placed an order for {order_size} units of {I[self.item_id]['NAME']}")
            self.env.process(provider.deliver(order_size, inventory))
            self.cal_procurement_cost(self)

    def cal_procurement_cost(self):
        self.daily_procurement_cost += self.purchase_cost * \
            I[self.item_id]["LOT_SIZE_ORDER"] + self.setup_cost

    def cal_daily_procurement_cost(self):
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
                print(
                    f"{self.env.now}: Stop {self.name} due to a shortage of input materials or WIPs")
                # Check again after 24 hours (1 day)
                yield self.env.timeout(24)
                # continue
            else:
                # Consuming input materials or WIPs and producing output WIP or Product
                processing_time = 24 / self.production_rate
                yield self.env.timeout(processing_time)
                print(f"{self.env.now}: Process {self.process_id} begins")
                for inven in self.input_inventories:
                    inven.level -= 1
                    print(
                        f"{self.env.now}: Inventory level of {I[inven.item_id]['NAME']}: {inven.level}")
                self.output_inventory.level += 1
                self.cal_processing_cost(processing_time)
                print(
                    f"{self.env.now}: A unit of {self.output['NAME']} has been produced")
                print(
                    f"{self.env.now}: Inventory level of {I[self.output_inventory.item_id]['NAME']}: {self.output_inventory.level}")

    def cal_processing_cost(self, processing_time):
        self.daily_production_cost += self.processing_cost * processing_time

    def cal_daily_production_cost(self):
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
                print(
                    f"{self.env.now}: {product_inventory.level} units of the product have been delivered to the customer")
                # yield self.env.timeout(DELIVERY_TIME)
                product_inventory.level -= order_size
                self.cal_selling_cost()
            print(
                f"{self.env.now}: Unable to deliver {num_shortages} units to the customer due to product shortage")
            # Check again after 24 hours (1 day)
            # yield self.env.timeout(24)
        # Delivering products to the customer
        else:
            product_inventory.level -= order_size
            print(
                f"{self.env.now}: {order_size} units of the product have been delivered to the customer")
            self.cal_selling_cost()

    def cal_selling_cost(self):
        self.daily_selling_cost += self.delivery_cost * \
            I[self.item_id]['DEMAND_QUANTITY'] + self.setup_cost

    def cal_daily_selling_cost(self):
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

    # Run the simulation
    for i in range(SIM_TIME*24):
        # Print the inventory level every 24 hours (1 day)
        if i % 24 == 0:
            if i != 0:
                # Calculate the cost models
                for inven in inventoryList:
                    inven.cal_inventory_cost()
                for production in productionList:
                    production.cal_daily_production_cost()
                for procurement in procurementList:
                    procurement.cal_daily_procurement_cost()
                sales.cal_daily_selling_cost()
            # Print the inventory level
            print(f"\nDAY {int(i/24)+1}")
            for inven in inventoryList:
                inven.level_over_time.append(inven.level)
                print(
                    f"[{I[inven.item_id]['NAME']}]  {inven.level}")
        env.run(until=i+1)

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
