import simpy
import numpy as np
from config import *
import random
import visualization

EventHoldingCost = []  # save the event holding cost
for i in range(SIM_TIME):
    num = []
    for j in range(len(I)):
        num.append([])
    EventHoldingCost.append(num)


class Inventory:
    def __init__(self, env, item_id, holding_cost, shortage_cost, initial_level):
        self.env = env
        self.item_id = item_id  # 0: product; others: WIP or raw material
        self.current_level = initial_level  # capacity=infinity
        self.unit_holding_cost = holding_cost/24  # $/unit*hour
        self.holding_cost_last_updated = 0.0
        self.daily_inven_cost = 0
        # self.unit_shortage_cost = shortage_cost
        # self.level_over_time = []  # Data tracking for inventory level
        # self.inventory_cost_over_time = []  # Data tracking for inventory cost
        # self.total_inven_cost = []

    def _cal_holding_cost(self, daily_events):
        holding_cost = self.current_level * self.unit_holding_cost * \
            (self.env.now - self.holding_cost_last_updated)
        self.holding_cost_last_updated = self.env.now
        daily_events.append(
            f"{self.env.now}: Daily holding cost of {I[self.item_id]['NAME']} has been updated: {holding_cost}")
        self.daily_inven_cost += holding_cost

    def update_inven_level(self, quantity_of_change, daily_events):
        self._cal_holding_cost(daily_events)
        self.current_level += quantity_of_change
        daily_events.append(
            f"{self.env.now}: Inventory level of {I[self.item_id]['NAME']}: {self.current_level}")

    def cal_inventory_cost(self, daily_events):
        if self.current_level > 0:
            self.inventory_cost_over_time.append(
                self.holding_cost * self.current_level)
        elif self.current_level < 0:
            self.inventory_cost_over_time.append(
                self.shortage_cost * abs(self.current_level))
        else:
            self.inventory_cost_over_time.append(0)
        daily_events.append(
            f"[Inventory Cost of {I[self.item_id]['NAME']}]  {self.inventory_cost_over_time[-1]}")


class Provider:
    def __init__(self, env, name, item_id):
        self.env = env
        self.name = name
        self.item_id = item_id

    def deliver(self, demand_size, inventory, daily_events):
        # Lead time
        yield self.env.timeout(I[self.item_id]["SUP_LEAD_TIME"] * 24)
        inventory.current_level += demand_size
        daily_events.append(
            f"{self.env.now}: {self.name} has delivered {demand_size} units of {I[self.item_id]['NAME']}")


class Procurement:
    def __init__(self, env, item_id, purchase_cost, setup_cost):
        self.env = env
        self.item_id = item_id
        self.unit_purchase_cost = purchase_cost
        self.unit_setup_cost = setup_cost
        self.daily_procurement_cost = 0
        # self.purchase_cost_over_time = []  # Data tracking for purchase cost
        # self.setup_cost_over_time = []  # Data tracking for setup cost

    def _cal_procurement_cost(self, order_size, daily_events):
        self.daily_procurement_cost += self.unit_purchase_cost * \
            order_size + self.unit_setup_cost
        daily_events.append(
            f"{self.env.now}: Daily procurement cost of {I[self.item_id]['NAME']} has been updated: {self.daily_procurement_cost}")

    def order(self, provider, inventory, agent, daily_events):
        while True:
            # Place an order to a provider
            yield self.env.timeout(I[self.item_id]["MANU_ORDER_CYCLE"] * 24)
            # THIS WILL BE AN ACTION OF THE AGENT
            # order_size = I[self.item_id]["LOT_SIZE_ORDER"]
            # order_size = agent.choose_action_tmp(inventory)
            order_size = random.choice(I[self.item_id]["LOT_SIZE_ORDER"])
            daily_events.append(
                f"{self.env.now}: Placed an order for {order_size} units of {I[self.item_id]['NAME']}")
            self.env.process(provider.deliver(
                order_size, inventory, daily_events))
            self._cal_procurement_cost(order_size, daily_events)

    def cal_daily_procurement_cost(self, daily_events):
        daily_events.append(
            f"[Daily procurement cost of {I[self.item_id]['NAME']}]  {self.daily_procurement_cost}")
        self.daily_procurement_cost = 0


class Production:
    def __init__(self, env, name, process_id, production_rate, output, input_inventories, qnty_for_input_item, output_inventory, processing_cost, process_stop_cost):
        self.env = env
        self.name = name
        self.process_id = process_id
        self.production_rate = production_rate
        self.output = output
        self.input_inventories = input_inventories
        self.qnty_for_input_item = qnty_for_input_item
        self.output_inventory = output_inventory
        self.unit_processing_cost = processing_cost
        # self.processing_cost_over_time = []  # Data tracking for processing cost
        self.unit_process_stop_cost = process_stop_cost
        self.daily_production_cost = 0

    def _cal_processing_cost(self, processing_time, daily_events):
        processing_cost = self.unit_processing_cost * processing_time
        self.daily_production_cost += processing_cost
        daily_events.append(
            f"{self.env.now}: Daily production cost of {self.name} has been updated: {processing_cost}")

    def process(self, daily_events):
        while True:
            # Check the current state if input materials or WIPs are available
            shortage_check = False
            for inven, input_qnty in zip(self.input_inventories, self.qnty_for_input_item):
                if inven.current_level < input_qnty:  # SHORTAGE
                    shortage_check = True
            if shortage_check:
                self.daily_production_cost += self.unit_process_stop_cost
                daily_events.append(
                    f"{self.env.now}: Stop {self.name} due to a shortage of input materials or WIPs")
                daily_events.append(
                    f"{self.env.now}: Process stop cost : {self.unit_process_stop_cost}")
                # Check again next day
                yield self.env.timeout(24 - (self.env.now % 24))
                # continue
            else:
                # Consuming input materials or WIPs and producing output WIP or Product
                daily_events.append(
                    f"{self.env.now}: Process {self.process_id} begins")
                # Update the inventory level for input items
                for inven, input_qnty in zip(self.input_inventories, self.qnty_for_input_item):
                    inven.update_inven_level(-input_qnty, daily_events)
                # Processing time has elapsed
                processing_time = 24 / self.production_rate
                yield self.env.timeout(processing_time)
                daily_events.append(
                    f"{self.env.now}: A unit of {self.output['NAME']} has been produced")
                # Update the inventory level for the output item
                self.output_inventory.update_inven_level(1, daily_events)
                # Calculate the processing cost
                self._cal_processing_cost(processing_time, daily_events)

    def cal_daily_production_cost(self, daily_events):
        daily_events.append(
            f"[Daily production cost of {self.name}]  {self.daily_production_cost}")
        self.daily_production_cost = 0


class Sales:
    def __init__(self, env, item_id, delivery_cost, setup_cost, due_date):
        self.env = env
        self.item_id = item_id
        self.unit_delivery_cost = delivery_cost
        self.unit_setup_cost = setup_cost
        self.due_date = due_date
        # self.selling_cost_over_time = []  # Data tracking for selling cost
        self.daily_selling_cost = 0
        # self.loss_cost = 0

    def _cal_selling_cost(self, demand_size, daily_events):
        self.daily_selling_cost += self.unit_delivery_cost * demand_size + self.setup_cost
        daily_events.append(
            f"{self.env.now}: Daily selling cost of {I[self.item_id]['NAME']} has been updated: {self.daily_selling_cost}")

    def delivery(self, item_id, demand_size, product_inventory, daily_events):
        # Lead time
        yield self.env.timeout(self.due_date * 24)
        # BACKORDER: Check if products are available
        if product_inventory.current_level < demand_size:
            num_shortages = abs(product_inventory.level - demand_size)
            if product_inventory.current_level > 0:
                daily_events.append(
                    f"{self.env.now}: {product_inventory.current_level} units of the product have been delivered to the customer")
                # yield self.env.timeout(DELIVERY_TIME)
                product_inventory.current_level -= demand_size
                self._cal_selling_cost()
            self.loss_cost = I[item_id]["BACKORDER_COST"] * num_shortages
            daily_events.append(f"[Cost of Loss] {self.loss_cost}")
            daily_events.append(
                f"{self.env.now}: Unable to deliver {num_shortages} units to the customer due to product shortage")
            # Check again after 24 hours (1 day)
            # yield self.env.timeout(24)
        # Delivering products to the customer
        else:
            product_inventory.update_inven_level(-demand_size, daily_events)
            daily_events.append(
                f"{self.env.now}: {demand_size} units of the product have been delivered to the customer")
            self._cal_selling_cost(demand_size, daily_events)

    def cal_daily_selling_cost(self, daily_events):
        daily_events.append(
            f"[Daily selling cost of  {I[self.item_id]['NAME']}]  {self.daily_selling_cost}")
        self.daily_selling_cost = 0


class Customer:
    def __init__(self, env, name, item_id):
        self.env = env
        self.name = name
        self.item_id = item_id
        self.order_history = []

    def order(self, sales, product_inventory, daily_events):
        while True:
            yield self.env.timeout(I[self.item_id]["CUST_ORDER_CYCLE"] * 24)
            # THIS WILL BE A RANDOM VARIABLE
            order_size = I[self.item_id]["DEMAND_QUANTITY"]
            self.order_history.append(order_size)
            daily_events.append(
                f"{self.env.now}: The customer has placed an order for {order_size} units of {I[self.item_id]['NAME']}")
            self.env.process(sales.delivery(
                self.item_id, order_size, product_inventory, daily_events))
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


def create_env(I, P, daily_events):
    # Create a SimPy environment
    simpy_env = simpy.Environment()
    # Create an inventory for each item
    inventoryList = []
    for i in I.keys():
        inventoryList.append(
            Inventory(simpy_env, i, I[i]["HOLD_COST"], I[i]["SHORTAGE_COST"], I[i]["INIT_LEVEL"]))
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
    # Create managers for manufacturing process, procurement process, and delivery process
    sales = Sales(simpy_env, customer.item_id,
                  I[0]["DELIVERY_COST"], I[0]["SETUP_COST_PRO"], I[0]["DUE_DATE"])
    productionList = []
    for i in P.keys():
        output_inventory = inventoryList[P[i]["OUTPUT"]["ID"]]
        input_inventories = []
        for j in P[i]["INPUT_TYPE_LIST"]:
            input_inventories.append(inventoryList[j["ID"]])
        productionList.append(Production(simpy_env, "PROCESS_"+str(i), P[i]["ID"],
                                         P[i]["PRODUCTION_RATE"], P[i]["OUTPUT"], input_inventories, P[i]["QNTY_FOR_INPUT_ITEM"], output_inventory, P[i]["PROCESS_COST"], P[i]["PROCESS_STOP_COST"]))

    return simpy_env, inventoryList, procurementList, productionList, sales, customer, providerList, daily_events


def simpy_event_processes(agent, simpy_env, inventoryList, procurementList, productionList, sales, customer, providerList, daily_events, I):
    # Event processes for SimPy simulation
    simpy_env.process(customer.order(
        sales, inventoryList[I[0]["ID"]], daily_events))  # Customer
    for production in productionList:  # Processes
        simpy_env.process(production.process(daily_events))
    for i in range(len(providerList)):  # Procurements
        simpy_env.process(procurementList[i].order(
            providerList[i], inventoryList[providerList[i].item_id], agent, daily_events))


def cal_cost(inventoryList, procurementList, productionList, sales, total_cost_per_day, daily_events):
    # Calculate the cost models
    for inven in inventoryList:
        inven.cal_inventory_cost(daily_events)
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
