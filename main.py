import environment as env
import numpy as np
import random
from visualization import *
from config import *
from DQN import *


def main():
    # 코드에 들어가는 옵션값
    total_cost_per_day = []

    # Initialize the simulation environment
    daily_events = []
    total_reward = 0  # 리워드 초기화
    simpy_env, inventoryList, procurementList, productionList, sales, customer, providerList, daily_events = env.create_env(
        I, P, daily_events)

    # Initialize the DQN agent
    state = np.array([inven.current_level for inven in inventoryList]
                     )  # Get the inventory levels
    state_size = len(inventoryList)  # Number of inventories
    agent = DQNAgent(state_size, action_space, discount_factor,
                     epsilon_greedy, epsilon_min, epsilon_decay,
                     learning_rate, max_memory_size, target_update_frequency)
    episode_done = False
    total_rewards, losses = [], []
    env.simpy_event_processes(agent, simpy_env, inventoryList, procurementList,
                              productionList, sales, customer, providerList, daily_events, I)

    # Print the list of items and processes
    print("\nItem list")
    for i in I.keys():
        print(f"ITEM {i}: {I[i]['NAME']}")
    print("\nProcess list")
    for i in P.keys():
        print(f"Output of PROCESS {i}: {P[i]['OUTPUT']['NAME']}")
    print("Number of Inventories: ", len(inventoryList))
    print("Number of Providers: ", len(providerList))

    total_cost = 0
    for episode in range(EPISODES):

        for i in range(SIM_TIME*24):  # i: hourly time step
            simpy_env.run(until=i+1)  # Run the simulation until the next hour

            if (i+1) % 24 == 0:  # Daily time step
                # Print the simulation log every 24 hours (1 day)
                if PRINT_SIM_EVENTS:
                    print(f"\nDay {(i+1) // 24}:")
                    for log in daily_events:
                        print(log)
                daily_events.clear()

                # Calculate the cost models
                daily_total_cost = 0
                for inven in inventoryList:
                    daily_total_cost += inven.daily_inven_cost
                    inven.daily_inven_cost = 0
                for production in productionList:
                    daily_total_cost += production.daily_production_cost
                    production.daily_production_cost = 0
                for procurement in procurementList:
                    daily_total_cost += procurement.daily_procurement_cost
                    procurement.daily_procurement_cost = 0
                daily_total_cost += sales.daily_selling_cost
                sales.daily_selling_cost = 0
                print("[Daily Total Cost] ", daily_total_cost)
                total_cost += daily_total_cost
        print("\n[Total Cost] ", total_cost)

        # Initialize the simulation environment
        total_reward = 0
        simpy_env, inventoryList, procurementList, productionList, sales, customer, providerList, daily_events = env.create_env(
            I, P, daily_events)

        '''
        if PRINT_DQN:
            print(
                "_________________________________________________________done")
        '''
        total_rewards.append(total_reward)
        print(f'Episode: {episode}/{EPISODES}, Total Reward: {total_reward}, Eps: {agent.epsilon:.2f}, Loss: {np.mean(losses):.5f}, Memory: {len(agent.memory)}')

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
    '''
    if SPECIFIC_HOLDING_COST:
        print(EventHoldingCost)
    
    #visualization
    if VISUAL :
        cost_list=[]#inventory_cost by id   id -> day 순으로 리스트 생성  전체 id 별로 저장되어 있는 list
        level_list=[]#inventory_level by id
        item_name_list=[]
        total_cost_per_day = env.cal_cost(inventoryList, productionList, procurementList,sales)
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


if __name__ == "__main__":
    main()
