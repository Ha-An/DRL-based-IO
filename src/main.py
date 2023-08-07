import simpy
import random
import visualization
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
# DUE_DATE: Due date of customer order [days]
# BACKORDER_COST: Backorder cost of products or WIP [$/unit]

COST_VALID = False
VISUAL = False

I = {0: {"ID": 0, "TYPE": "Product",      "NAME": "PRODUCT",          "CUST_ORDER_CYCLE": 7, "DEMAND_QUANTITY": 21, "MANU_LEAD_TIME": 7,                      "HOLD_COST": 5, "SHORTAGE_COST": 10,                     "SETUP_COST_PRO": 50, "DELIVERY_COST": 10, "DUE_DATE": 2, "BACKORDER_COST": 5},
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
SIM_TIME = 3  # [days]
INITIAL_INVENTORY = 30  # [units]
RAW_MATERIALS = 2 #Max number of process materials
INV_COST=np.zeros((SIM_TIME,24,1+ RAW_MATERIALS*2+1))  
                    #(day,hours,Product + RawMaterials * Number of Process + WIP)
EVENT_HOLDING_COST = []  # save the event holding cost
for i in range(SIM_TIME):
    num = []
    for j in range(len(I)):
        num.append([])
    EVENT_HOLDING_COST.append(num)
    
    
print(EVENT_HOLDING_COST)
#inventory(env, i, I[i]["HOLD_COST"], I[i]["SHORTAGE_COST"]))
class Inventory:
    def __init__(self, env, item_id, holding_cost, shortage_cost):
        self.item_id = item_id  # 0: product; others: WIP or raw material
        self.level = INITIAL_INVENTORY  # capacity=infinity
        self.holding_cost = holding_cost  # $/unit*day
        self.shortage_cost = shortage_cost
        self.level_over_time = []  # Data tracking for inventory level
        self.inventory_cost_over_time = []  # Data tracking for inventory cost
        self.total_inven_cost= []
    
        
        
    def cal_inventory_cost(self):
        
        if self.level > 0:
            self.inventory_cost_over_time.append(
                self.holding_cost * self.level)
        elif self.level < 0:
            self.inventory_cost_over_time.append(
                self.shortage_cost * abs(self.level))
        else:
            self.inventory_cost_over_time.append(0)
        self.total_inven_cost = self.inventory_cost_over_time
        print(
            f"[Inventory Cost of {I[self.item_id]['NAME']}]  {self.inventory_cost_over_time[-1]}")
        print( f"[Inventory Cost of {I[self.item_id]['NAME']}]  {self.inventory_cost_over_time}")

    

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
            self.cal_procurement_cost()

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
        self.temp=9999


        
    
       #can change variables
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
                i=0
                for inven in self.input_inventories:
                    inven.level -= 1
                    
                    print(
                        f"{self.env.now}: Inventory level of {I[inven.item_id]['NAME']}: {inven.level}")
                    print(f"{self.env.now}: Holding cost of {I[inven.item_id]['NAME']}: {round((inven.level*I[inven.item_id]['HOLD_COST']/24*self.production_rate),2)}")
                    INV_COST[int(self.env.now/24)][int(self.env.now)%24][RAW_MATERIALS * self.process_id + i + 1]=round((inven.level*I[inven.item_id]['HOLD_COST']/24*self.production_rate),2)
                    i=i+1
                    print('value:',round((inven.level*I[inven.item_id]['HOLD_COST']/24*self.production_rate),2))
                    EVENT_HOLDING_COST[int(self.env.now/24)][inven.item_id].append(round((inven.level*I[inven.item_id]['HOLD_COST']/24*self.production_rate),2))
                    self.temp=inven.item_id
 
            
                self.output_inventory.level += 1
                self.cal_processing_cost(processing_time)
                
                print(
                    f"{self.env.now}: A unit of {self.output['NAME']} has been produced")
                print(
                    f"{self.env.now}: Inventory level of {I[self.output_inventory.item_id]['NAME']}: {self.output_inventory.level}")
                print(
                    f"{self.env.now}: Holding cost of {I[self.output_inventory.item_id]['NAME']}: {round((self.output_inventory.level*I[self.output_inventory.item_id]['HOLD_COST']/24*self.production_rate),2)}")
                            
                INV_COST[int(self.env.now/24)][int(self.env.now)%24][-1]=(round((self.output_inventory.level*I[self.output_inventory.item_id]['HOLD_COST']/24*self.production_rate),2))
                EVENT_HOLDING_COST[int(self.env.now/24)][-1].append(round((self.output_inventory.level*I[self.output_inventory.item_id]['HOLD_COST']/24*self.production_rate),2))
    
    def cal_processing_cost(self, processing_time):
        self.daily_production_cost += self.processing_cost * processing_time

    def cal_daily_production_cost(self):
        print(
            f"[Daily production cost of {self.name}]  {self.daily_production_cost}")
        INV_COST[int(self.env.now/24)][int(self.env.now)%24][0]=(round(self.daily_production_cost,2))
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
        print('inventory level-1',product_inventory.level)
        print('order size',order_size)
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
            
    def cost_of_loss(self,item_id, order_size, product_inventory): #due to not enough production
        if I[item_id]["MANU_LEAD_TIME"] > I[item_id]["DUE_DATE"]:
            if product_inventory.level < order_size:
                num_shortages = abs(product_inventory.level - order_size)
                loss_cost = I[item_id]["BACKORDER_COST"] * num_shortages
                print(f"[Cost of Loss] {loss_cost}")
            else:
                loss_cost = 0
                print(f"[Cost of Loss] : {loss_cost}")
            return loss_cost

    def cal_selling_cost(self,cost):
        self.daily_selling_cost += self.delivery_cost * \
            I[self.item_id]['DEMAND_QUANTITY'] + self.setup_cost + cost

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
            self.env.process(sales.delivery(self.item_id, order_size, product_inventory))
            #self.env.process(sales.cost_of_loss(self.item_id, order_size, product_inventory))
            
    
    
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
def cal_cost(inventoryList, productionList, procurementList,sales):
    total_cost_per_day=[]
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

                # Reset daily cost variables
                for inven in inventoryList:
                    inven.inventory_cost_over_time = []
                for production in productionList:
                    production.daily_production_cost = 0
                for procurement in procurementList:
                    procurement.daily_procurement_cost = 0
                sales.daily_selling_cost = 0
                
                print(total_cost_per_day)
    return total_cost_per_day
            
            
                
        

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
                cost=sales.cost_of_loss(0,I[customer.item_id]['DEMAND_QUANTITY'],inventoryList[I[0]["ID"]])
                sales.cal_daily_selling_cost()
                
            
            # Print the inventory level
            print(f"\nDAY {int(i/24)+1}")
            for inven in inventoryList:
                inven.level_over_time.append(inven.level)
                if inven.level>=0:
                    print(
                    f"[{I[inven.item_id]['NAME']}]  {inven.level}")
                else:
                    print(
                    f"[{I[inven.item_id]['NAME']}]  0")
           
           
        env.run(until=i+1)
    
    print(EVENT_HOLDING_COST)
    

    if COST_VALID:
        print(INV_COST)
    
    
        
        
    #visualization
    if VISUAL :
        cost_list=[]#inventory_cost by id   id -> day 순으로 리스트 생성  전체 id 별로 저장되어 있는 list
        level_list=[]#inventory_level by id
        item_name_list=[]
        total_cost_per_day = cal_cost(inventoryList, productionList, procurementList,sales)
        total_cost_list = total_cost_per_day
        for i in I.keys():
            temp1=[]
            temp2=[]
            inventory_visualization = visualization.visualization(
                inventoryList[i])
            temp1,temp2=inventory_visualization.return_list()
            level_list.append(temp1)
            cost_list.append(temp2)
            item_name_list.append(I[i]['NAME'])
        inventory_visualization = visualization.visualization(None) # 필요하지 않으므로 None
        inventory_visualization.plot_inventory_graphs(level_list, cost_list,total_cost_list,item_name_list)
        
       
    
        

'''
#visualization
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))
    for i in I.keys():
        plt.plot(inventoryList[i].level_over_time, label=f"[{inventoryList[i]['NAME']}]")
    plt.xlabel('time[days]')
    plt.ylabel('inventory')    
    plt.legend() 
    plt.grid(True)
    plt.show()
       

    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))
    for i in I.keys():
        plt.plot(inventoryList[i].inventory_cost_over_time, label=i)
    plt.xlabel('time[days]')
    plt.ylabel('inventory')    
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))
    for i in I.keys():
        plt.plot(inventoryList[i].inventory_cost_over_time, label=i)
    plt.xlabel('time[hours]')
    plt.ylabel('inventory')    
    plt.legend()
    plt.grid(True)
    plt.show()
'''    

if __name__ == "__main__":
    main()
