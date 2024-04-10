from math import e
import simpy
import numpy as np
from config import *  # Assuming this imports necessary configurations
import random
from log import *  # Assuming this imports necessary logging functionalities


class Inventory:
    def __init__(self, env, item_id, holding_cost):
        # Initialize inventory object
        self.env = env
        self.item_id = item_id  # 0: product; others: WIP or material
        self.on_hand_inventory = INIT_LEVEL  # Initial inventory level
        self.in_transition_inventory = 0  # Inventory in transition (e.g., being delivered)
        self.capacity_limit = INVEN_LEVEL_MAX  # Maximum capacity of the inventory
        # Daily inventory report template
        self.daily_inven_report = [f"Day {self.env.now // 24}", I[self.item_id]['NAME'],
                                   I[self.item_id]['TYPE'], self.on_hand_inventory, 0, 0, 0]  # 0 0 0 이 뭘뜻하는지 모르겠음
        # Unit holding cost per hour
        self.unit_holding_cost = holding_cost / 24
        self.holding_cost_last_updated = 0.0  # Last time holding cost was updated

    def update_demand_quantity(self, demand_qty, daily_events):
        """
        Update the demand quantity and log the event.
        """
        DEMAND_LOG.append(demand_qty)
        daily_events.append(
            f"{self.env.now}: Customer order of {I[0]['NAME']}                                : {I[0]['DEMAND_QUANTITY']} units ") #현재시간기준 

    def update_inven_level(self, quantity_of_change, inven_type, daily_events):
        """
        Update the inventory level based on the quantity of change and log the event.
        """
        if I[self.item_id]["TYPE"] == "Material":
            if quantity_of_change < 0 and inven_type == "ON_HAND":  #재고가 감소하고 on hand일때 
                self._update_report(quantity_of_change)
            elif inven_type == "IN_TRANSIT" and quantity_of_change > 0:  #재고가 증가하는데 타입이 이동중일때 
                self.change_qty = quantity_of_change
                self._update_report(quantity_of_change)
        else:
            self._update_report(quantity_of_change)

        if inven_type == "ON_HAND":
            # Update on-hand inventory
            Cost.cal_cost(self, "Holding cost")
            self.on_hand_inventory += quantity_of_change
            
            # Check if inventory exceeds capacity limit
            if self.on_hand_inventory > self.capacity_limit:
                daily_events.append(
                    f"{self.env.now}: Due to the upper limit of the inventory, {I[self.item_id]['NAME']} is wasted: {self.on_hand_inventory - self.capacity_limit}") # 재고가 꽉참 
                self.on_hand_inventory = self.capacity_limit
            # Check if inventory goes negative
            if self.on_hand_inventory < 0:  #재고부족 상황 ?????
                daily_events.append(
                    f"{self.env.now}: Shortage of {I[self.item_id]['NAME']}: {self.capacity_limit - self.on_hand_inventory}")  # ex 50-(-2) =52 부족 ??
                self.on_hand_inventory = 0
            
            self.holding_cost_last_updated = self.env.now

        elif inven_type == "IN_TRANSIT":
            # Update in-transition inventory
            self.in_transition_inventory += quantity_of_change

    def _update_report(self, quantity_of_change):
        """
        Update the daily inventory report based on the quantity of change.
        """
        if quantity_of_change > 0:
            self.daily_inven_report[4] += quantity_of_change
        elif quantity_of_change == 0:
            pass
        else:
            self.daily_inven_report[5] -= quantity_of_change


class Supplier:
    def __init__(self, env, name, item_id):
        # Initialize supplier object
        self.env = env
        self.name = name
        self.item_id = item_id

    def deliver_to_manufacturer(self, procurement, material_qty, material_inventory, daily_events):
        """
        Deliver materials to the manufacturer after a certain lead time.
        """
        I[self.item_id]["SUP_LEAD_TIME"] = random.randint(0, 5) #공급자부터 제조업체까지 배송하는데 걸리는 시간을 랜덤으로 
        lead_time = I[self.item_id]["SUP_LEAD_TIME"]
        # Log the delivery event with lead time
        daily_events.append(
            f"{self.env.now}: {I[self.item_id]['NAME']} will be delivered at {lead_time} days after         : {I[self.item_id]['LOT_SIZE_ORDER']} units")

        # Wait for the lead time
        yield self.env.timeout(lead_time * 24)  # 실제 배송이 될때까지 대기함 

        # Receive materials by calling the receive_materials method of procurement
        procurement.receive_materials(
            material_qty, material_inventory, daily_events) #원자재 수령 (양을 얻도 인벤토리에 추가)
        


class Procurement:
    def __init__(self, env, item_id, purchase_cost, setup_cost):
        self.env = env
        self.item_id = item_id
        self.unit_purchase_cost = purchase_cost
        self.unit_setup_cost = setup_cost
        self.order_qty = 0  # Initialize order quantity

    def receive_materials(self, material_qty, material_inventory, daily_events):
        """
        Process the receipt of materials and update inventory.
        """
        daily_events.append(
            f"==============={I[self.item_id]['NAME']} Delivered ===============")

        # Update in-transition inventory (reduce it)
        material_inventory.update_inven_level(
            -material_qty, "IN_TRANSIT", daily_events)  # 인벤토리  트랜지션 마이너스 
        # Update on-hand inventory (increase it)
        material_inventory.update_inven_level(
            material_qty, "ON_HAND", daily_events)  # 인벤토리에서 온핸드 증가 
        daily_events.append(
            f"{self.env.now}: {I[self.item_id]['NAME']} has delivered                             : {material_qty} units ")  # Record when Material provide

    def order_material(self, supplier, inventory, daily_events):  #공급자에게 원자재 주문 
        """
        Place orders for materials to the supplier.
        """
        yield self.env.timeout(self.env.now)  # Wait for the next order cycle   ???? 주문주기의 시작??(24)
        while True:
            daily_events.append(
                f"==============={I[self.item_id]['NAME']}\'s Inventory ===============")

            # Set the order size based on LOT_SIZE_ORDER and reorder level
            #I[self.item_id]["LOT_SIZE_ORDER"] = ORDER_QTY
            order_size = I[self.item_id]["LOT_SIZE_ORDER"] #주문크기 정하기 
            if order_size > 0 and inventory.on_hand_inventory < REORDER_LEVEL:  #리오더 =10 
                daily_events.append(
                    f"{self.env.now}: The Procurement ordered {I[self.item_id]['NAME']}: {I[self.item_id]['LOT_SIZE_ORDER']}  units  ")
                self.order_qty = order_size
                # Update in-transition inventory
                inventory.update_inven_level(
                    order_size, "IN_TRANSIT", daily_events)
                # Calculate and log order cost
                Cost.cal_cost(self, "Order cost")
                # Initiate the delivery process by calling deliver_to_manufacturer method of the supplier
                self.env.process(supplier.deliver_to_manufacturer(
                    self, order_size, inventory, daily_events)) # 공급자에게 배송을 요청 리드타임후 배송
                # Record in-transition inventory
                daily_events.append(
                    f"{self.env.now}: {I[self.item_id]['NAME']}\'s In_transition_inventory                  : {inventory.in_transition_inventory} units ")
                # Record total inventory
                daily_events.append(
                    f"{self.env.now}: {I[self.item_id]['NAME']}\'s Total_Inventory                          : {inventory.in_transition_inventory+inventory.on_hand_inventory} units  ")
            yield self.env.timeout(I[self.item_id]["MANU_ORDER_CYCLE"] *  24)  # Wait for the next order cycle  ???? 다음주기까지 대기?


class Production:
    def __init__(self, env, name, process_id, production_rate, output, input_inventories, qnty_for_input_item, output_inventory, processing_cost, process_stop_cost):
        # Initialize production process
        self.env = env
        self.name = name
        self.process_id = process_id
        self.production_rate = production_rate
        self.output = output
        self.input_inventories = input_inventories
        self.qnty_for_input_item = qnty_for_input_item
        self.output_inventory = output_inventory
        self.processing_time = 24 / self.production_rate
        self.unit_processing_cost = processing_cost/self.processing_time

    def process_items(self, daily_events):
        """
        Simulate the production process.
        """
        while True:
            daily_events.append(
                "===============Process Phase===============")

            # Check if there's a shortage of input materials or WIPs
            shortage_check = False  #재고 또는 wip의 부족 검사 
            for inven, input_qnty in zip(self.input_inventories, self.qnty_for_input_item):
                if inven.on_hand_inventory < input_qnty:  #현재 재고량이 필요수량보다 적은지확인
                    shortage_check = True

            # Check if the output inventory is full
            inven_upper_limit_check = False  #출력 인벤의 용량 검사 
            if self.output_inventory.on_hand_inventory >= self.output_inventory.capacity_limit:
                inven_upper_limit_check = True

            if shortage_check:
                daily_events.append(
                    f"{self.env.now}: Stop {self.name} due to a shortage of input materials or WIPs")
                yield self.env.timeout(24 - (self.env.now % 24))  #작업을 하지않고 다음날 자정까지 대기
            elif inven_upper_limit_check:
                daily_events.append(
                    f"{self.env.now}: Stop {self.name} due to the upper limit of the inventory. The output inventory is full")
                yield self.env.timeout(24 - (self.env.now % 24))
            else:
                daily_events.append(
                    f"{self.env.now}: Process {self.process_id} begins")

                # Consume input materials or WIPs
                for inven, input_qnty in zip(self.input_inventories, self.qnty_for_input_item):
                    inven.update_inven_level(-input_qnty,
                                             "ON_HAND", daily_events)

                # Process items (consume time)
                Cost.cal_cost(self, "Process cost")
                yield self.env.timeout(self.processing_time) #공정에 소요되는 시간만큼 전진시킴
                daily_events.append(
                    "===============Result Phase================")
                daily_events.append(
                    f"{self.env.now}: {self.output['NAME']} has been produced                         : 1 units")

                # Update the output inventory
                self.output_inventory.update_inven_level(
                    1, "ON_HAND", daily_events)


class Sales:
    def __init__(self, env, item_id, delivery_cost, setup_cost, shortage, due_date):
        # Initialize sales process
        self.env = env
        self.item_id = item_id
        self.due_date = due_date
        self.unit_delivery_cost = delivery_cost
        self.unit_setup_cost = setup_cost
        self.unit_shortage_cost = shortage
        self.delivery_item = 0
        self.num_shortages = 0

    def _deliver_to_cust(self, demand_size, product_inventory, daily_events):
        """
        Deliver products to customers and handle shortages if any.
        """
        yield self.env.timeout(I[self.item_id]["DUE_DATE"] * 24)  #대기기한까지 대기 
        # Check if products are available for delivery
        if product_inventory.on_hand_inventory < demand_size:
            # Calculate the shortage
            self.num_shortages = abs(
                product_inventory.on_hand_inventory - demand_size)  #abs는 절댓값 반환 
            # If there are some products available, deliver them first
            if product_inventory.on_hand_inventory > 0:  #일부라도 배송 
                self.delivery_item = product_inventory.on_hand_inventory
                daily_events.append(
                    f"{self.env.now}: PRODUCT have been delivered to the customer       : {product_inventory.on_hand_inventory} units ")
                # Update inventory
                product_inventory.update_inven_level(
                    -product_inventory.on_hand_inventory, 'ON_HAND', daily_events)

            # Calculate and log shortage cost
            Cost.cal_cost(self, "Shortage cost")
            daily_events.append(
                f"{self.env.now}: Unable to deliver {self.num_shortages} units to the customer due to product shortage")
        else:
            # Deliver products to the customer
            self.delivery_item = demand_size
            product_inventory.update_inven_level(
                -demand_size, 'ON_HAND', daily_events)
            daily_events.append(
                f"{self.env.now}: PRODUCT have been delivered to the customer       : {demand_size} units  ")

    def receive_demands(self, demand_qty, product_inventory, daily_events):
        """
        Receive demands from customers and initiate the delivery process.
        """
        # Update demand quantity in inventory
        product_inventory.update_demand_quantity(demand_qty, daily_events)  #인벤토리
        # Initiate delivery process
        self.env.process(self._deliver_to_cust(
            demand_qty, product_inventory, daily_events))  #simpy 프로세스 시작 


class Customer:
    def __init__(self, env, name, item_id):
        # Initialize customer object
        self.env = env
        self.name = name
        self.item_id = item_id

    def order_product(self, sales, product_inventory, daily_events):  #고객이 제품 주문 
        """
        Place orders for products to the sales process.
        """
        yield self.env.timeout(self.env.now)  # Wait for the next order cycle
        while True:
            # Generate a random demand quantity
            I[0]["DEMAND_QUANTITY"] = random.randint(
                DEMAND_QTY_MIN, DEMAND_QTY_MAX)  #랜덤한 수요량 
            demand_qty = I[0]["DEMAND_QUANTITY"]
            # Receive demands and initiate delivery process
            sales.receive_demands(demand_qty, product_inventory, daily_events)
            # Wait for the next order cycle
            yield self.env.timeout(I[0]["CUST_ORDER_CYCLE"] * 24)


class Cost:
    # Class for managing costs in the simulation
    def cal_cost(instance, cost_type):
        """
        Calculate and log different types of costs.
        """
        if cost_type == "Holding cost":
            # Calculate holding cost
            DAILY_COST_REPORT[cost_type] += instance.unit_holding_cost * instance.on_hand_inventory * (
                instance.env.now - instance.holding_cost_last_updated)
        elif cost_type == "Process cost":
            # Calculate processing cost
            DAILY_COST_REPORT[cost_type] += instance.unit_processing_cost * instance.processing_time
        elif cost_type == "Delivery cost":
            # Calculate delivery cost
            DAILY_COST_REPORT[cost_type] += instance.unit_delivery_cost * instance.delivery_item + instance.unit_setup_cost
        elif cost_type == "Order cost":
            # Calculate order cost
            DAILY_COST_REPORT[cost_type] += instance.unit_purchase_cost * instance.order_qty + instance.unit_setup_cost
        elif cost_type == "Shortage cost":
            # Calculate shortage cost
            DAILY_COST_REPORT[cost_type] += instance.unit_shortage_cost * instance.num_shortages

    def update_cost_log(inventoryList):
        """
        Update the cost log at the end of each day.
        """
        COST_LOG.append(0)  #비용로그 초기화
        # Update holding cost
        for inven in inventoryList:
            DAILY_COST_REPORT['Holding cost'] += inven.unit_holding_cost * inven.on_hand_inventory * (
                inven.env.now - inven.holding_cost_last_updated)
            inven.holding_cost_last_updated = inven.env.now #홀딩코스트가 업데이트되는 시간 

        # Update daily total cost
        for key in DAILY_COST_REPORT.keys():
            COST_LOG[-1] += DAILY_COST_REPORT[key]  #마지막요소의 비용을 더함 
        
        return COST_LOG[-1]

    def clear_cost():
        """
        Clear the daily cost report.
        """
        # Clear daily report
        for key in DAILY_COST_REPORT.keys():
            DAILY_COST_REPORT[key] = 0  #비용 보고서 초기화


def create_env(I, P, daily_events):  #시뮬레이션 환경 생성 필요한 조건들을 초기화
    # Function to create the simulation environment and necessary objects
    simpy_env = simpy.Environment()  # Create a SimPy environment

    # Create an inventory for each item
    inventoryList = []
    for i in I.keys():
        inventoryList.append(
            Inventory(simpy_env, i, I[i]["HOLD_COST"])) #인벤토리 객체는 환경 아이템  보유비용 인자로 받음  생성된 인벤토리 객체는 inventoryList 추가

    # Create stakeholders (Customer, Suppliers)
    customer = Customer(simpy_env, "CUSTOMER", I[0]["ID"]) # 커스터머의 인스턴스 생성 고객의 역할을 수행 특정아이템 주문 

    supplierList = [] #공급자
    procurementList = [] #조달 
    for i in I.keys():
        if I[i]["TYPE"] == 'Material': #아이템 딕셔너리 I
            supplierList.append(Supplier(simpy_env, "SUPPLIER_" + str(i), i))
            procurementList.append(Procurement(
                simpy_env, I[i]["ID"], I[i]["PURCHASE_COST"], I[i]["ORDER_COST_TO_SUP"]))

    # Create managers for manufacturing process, procurement process, and delivery process
    sales = Sales(simpy_env, customer.item_id,
                  I[0]["DELIVERY_COST"], I[0]["SETUP_COST_PRO"], I[0]["SHORTAGE_COST_PRO"], I[0]["DUE_DATE"])
    productionList = []
    for i in P.keys():
        output_inventory = inventoryList[P[i]["OUTPUT"]["ID"]] #제조공정 P
        input_inventories = []
        for j in P[i]["INPUT_TYPE_LIST"]:
            input_inventories.append(inventoryList[j["ID"]])
        productionList.append(Production(simpy_env, "PROCESS_" + str(i), P[i]["ID"],
                                         P[i]["PRODUCTION_RATE"], P[i]["OUTPUT"], input_inventories, P[i]["QNTY_FOR_INPUT_ITEM"], output_inventory, P[i]["PROCESS_COST"], P[i]["PROCESS_STOP_COST"]))

    return simpy_env, inventoryList, procurementList, productionList, sales, customer, supplierList, daily_events


def simpy_event_processes(simpy_env, inventoryList, procurementList, productionList, sales, customer, supplierList, daily_events, I):
    # Event processes for SimPy simulation
    for production in productionList:
        simpy_env.process(production.process_items(daily_events))
    for i in range(len(supplierList)):
        simpy_env.process(procurementList[i].order_material(
            supplierList[i], inventoryList[supplierList[i].item_id], daily_events))
    simpy_env.process(customer.order_product(
        sales, inventoryList[I[0]["ID"]], daily_events))


def update_daily_report(inventoryList):
    # Update daily reports for inventory
    day_report_list = []
    for inven in inventoryList:
        inven.daily_inven_report[-1] = inven.on_hand_inventory + inven.in_transition_inventory
        day_report_list.append(inven.daily_inven_report)
        inven.daily_inven_report = [f"Day {inven.env.now//24}", I[inven.item_id]['NAME'], I[inven.item_id]['TYPE'],
                                    inven.on_hand_inventory+inven.in_transition_inventory, 0, 0, 0]  # inventory report


def cap_current_state(inventoryList):
    # Function to capture the current state of the inventory
    state = np.array([inven.on_hand_inventory for inven in inventoryList])
    if STATE_DEMAND:
        # Include demand quantity in the state if required
        state = np.append(state, I[0]['DEMAND_QUANTITY'])
    return state
