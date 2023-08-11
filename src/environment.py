import simpy
import numpy as np
from config import *
import random
import visualization


class Inventory:
    def __init__(self, env, item_id, holding_cost, shortage_cost, initial_level):
        self.item_id = item_id  # 0: product; others: WIP or raw material
        self.level = initial_level  # capacity=infinity
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
        if Ver_simulation:
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
        if Ver_simulation:
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
            if Ver_simulation:
                print(
                    f"{self.env.now}: Placed an order for {order_size} units of {I[self.item_id]['NAME']}")
            self.env.process(provider.deliver(order_size, inventory))
            self.cal_procurement_cost()

    def cal_procurement_cost(self):
        self.daily_procurement_cost += self.purchase_cost * \
            I[self.item_id]["LOT_SIZE_ORDER"] + self.setup_cost

    def cal_daily_procurement_cost(self):
        if Ver_simulation:
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
                if Ver_simulation:
                    print(
                        f"{self.env.now}: Stop {self.name} due to a shortage of input materials or WIPs")
                # Check again after 24 hours (1 day)
                yield self.env.timeout(24)
                # continue
            else:
                # Consuming input materials or WIPs and producing output WIP or Product
                processing_time = 24 / self.production_rate
                yield self.env.timeout(processing_time)
                if Ver_simulation:
                    print(f"{self.env.now}: Process {self.process_id} begins")
                for inven in self.input_inventories:
                    inven.level -= 1
                    if Ver_simulation:
                        print(
                            f"{self.env.now}: Inventory level of {I[inven.item_id]['NAME']}: {inven.level}")
                self.output_inventory.level += 1
                self.cal_processing_cost(processing_time)
                if Ver_simulation:
                    print(
                        f"{self.env.now}: A unit of {self.output['NAME']} has been produced")
                    print(
                        f"{self.env.now}: Inventory level of {I[self.output_inventory.item_id]['NAME']}: {self.output_inventory.level}")

    def cal_processing_cost(self, processing_time):
        self.daily_production_cost += self.processing_cost * processing_time

    def cal_daily_production_cost(self):
        if Ver_simulation:
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
                if Ver_simulation:
                    print(
                        f"{self.env.now}: {product_inventory.level} units of the product have been delivered to the customer")
                # yield self.env.timeout(DELIVERY_TIME)
                product_inventory.level -= order_size
                self.cal_selling_cost()
            if Ver_simulation:
                print(
                    f"{self.env.now}: Unable to deliver {num_shortages} units to the customer due to product shortage")
            # Check again after 24 hours (1 day)
            # yield self.env.timeout(24)
        # Delivering products to the customer
        else:
            product_inventory.level -= order_size
            if Ver_simulation:
                print(
                    f"{self.env.now}: {order_size} units of the product have been delivered to the customer")
            self.cal_selling_cost()

    def cal_selling_cost(self):
        self.daily_selling_cost += self.delivery_cost * \
            I[self.item_id]['DEMAND_QUANTITY'] + self.setup_cost

    def cal_daily_selling_cost(self):
        if Ver_simulation:
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
            if Ver_simulation:
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


def create_env(I, P):
    simpy_env = simpy.Environment()

    # Create an inventory for each item
    inventoryList = []
    for i in I.keys():
        inventoryList.append(
            Inventory(simpy_env, i, I[i]["HOLD_COST"], I[i]["SHORTAGE_COST"], I[i]["INIT_LEVEL"]))
    if Ver_simulation:
        print("Number of Inventories: ", len(inventoryList))

    # Create stakeholders (Customer, Providers)
    customer = Customer(simpy_env, "CUSTOMER", I[0]["ID"])
    providerList = []
    procurementList = []
    for i in I.keys():
        # Create a provider and the corresponding procurement if the type of the item is Raw Material
        if I[i]["TYPE"] == 'Raw Material':
            providerList.append(Provider(simpy_env, "PROVIDER_"+str(i), i))
            procurementList.append(Procurement(
                simpy_env, I[i]["ID"], I[i]["PURCHASE_COST"], I[i]["SETUP_COST_RAW"]))
    if Ver_simulation:
        print("Number of Providers: ", len(providerList))

    # Create managers for manufacturing process, procurement process, and delivery process
    sales = Sales(simpy_env, customer.item_id,
                  I[0]["DELIVERY_COST"], I[0]["SETUP_COST_PRO"])
    productionList = []
    for i in P.keys():
        output_inventory = inventoryList[P[i]["OUTPUT"]["ID"]]
        input_inventories = []
        for j in P[i]["INPUT_LIST"]:
            input_inventories.append(inventoryList[j["ID"]])
        productionList.append(Production(simpy_env, "PROCESS_"+str(i), P[i]["ID"],
                                         P[i]["PRODUCTION_RATE"], P[i]["OUTPUT"], input_inventories, output_inventory, P[i]["PROCESS_COST"]))

    # Event processes for SimPy simulation
    simpy_env.process(customer.order(sales, inventoryList[I[0]["ID"]]))
    for production in productionList:
        simpy_env.process(production.process())
    for i in range(len(providerList)):
        simpy_env.process(procurementList[i].order(
            providerList[i], inventoryList[providerList[i].item_id]))

    return simpy_env, inventoryList, procurementList, productionList, sales, customer, providerList


def reset_env(inventoryList, procurementList, productionList, sales, customer, providerList):
    simpy_env = simpy.Environment()  # 환경초기화

    # Reset the inventory level for each item
    for i in I.keys():
        inventoryList[i].level = I[i]["INIT_LEVEL"]

    # Event processes for SimPy simulation
    simpy_env.process(customer.order(sales, inventoryList[I[0]["ID"]]))
    for production in productionList:
        simpy_env.process(production.process())
    for i in range(len(providerList)):
        simpy_env.process(procurementList[i].order(
            providerList[i], inventoryList[providerList[i].item_id]))
    state = np.array([inven.level for inven in inventoryList]
                     )  # Get the initial inventory levels
    state = state.reshape(1, len(inventoryList))

    return simpy_env, state


def cal_cost(inventoryList, procurementList, productionList, sales, total_cost_per_day):
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
