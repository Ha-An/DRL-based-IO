import environment as env
import numpy as np
import random
from visualization import *
from config import *
from DQN import *

def take_action(action_space, action, simpy_env, inventoryList, total_cost_per_day, I):
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
    simpy_env.run(until=simpy_env.now + 24)

    # Calculate the next state after the actions are taken
    next_state = np.array([inven.level for inven in inventoryList])
    next_state = next_state.reshape(1, len(inventoryList))

    # Calculate the reward and whether the simulation is done
    # You need to define this function based on your specific reward policy
    reward = -total_cost_per_day[-1]
    # Terminate the episode if the simulation time is reached
    done = (simpy_env.now >= SIM_TIME * 24)

    return next_state, reward, done

def main():  

    # Print the list of items and processes
    print("\nItem list")
    for i in I.keys():
        print(f"ITEM {i}: {I[i]['NAME']}")
    print("\nProcess list")
    for i in P.keys():
        print(f"Output of PROCESS {i}: {P[i]['OUTPUT']['NAME']}")

    simpy_env, inventoryList, procurementList, productionList, sales, customer, providerList = env.create_env(I, P)

    # 코드에 들어가는 옵션값
    total_cost_per_day = []
    # Run the simulation
    state = np.array([inven.level for inven in inventoryList])  # Get the inventory levels
    state_size = len(inventoryList)  # Number of inventories
    state = state.reshape(1, state_size)
    agent = DQNAgent(state_size, action_space)
    done = (simpy_env.now >= SIM_TIME * 24)
    total_rewards, losses = [], []
    total_reward = 0
    for episode in range(EPISODES):
        for i in range(SIM_TIME*24+24):
            # Print the inventory level every 24 hours (1 day)
            if i % 24 == 0:
                if i != 0:
                    if Ver_print:
                        print("day", i/24)
                        
                    env.cal_cost(inventoryList, procurementList, productionList, sales, total_cost_per_day)
                       
                    action = agent.choose_action(state)
                    next_state, reward, done = take_action(
                        action_space, action, simpy_env, inventoryList, total_cost_per_day, I)
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
                         
                        simpy_env, state = env.reset_env(inventoryList, procurementList, productionList, sales, customer, providerList)
      
                        break
                    
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
        

if __name__ == "__main__":
    main()