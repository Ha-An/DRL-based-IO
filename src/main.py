import environment as env
import numpy as np
import random
from visualization import *
from config import *
from DQN import *


def main():

    # Print the list of items and processes
    print("\nItem list")
    for i in I.keys():
        print(f"ITEM {i}: {I[i]['NAME']}")
    print("\nProcess list")
    for i in P.keys():
        print(f"Output of PROCESS {i}: {P[i]['OUTPUT']['NAME']}")

    simpy_env, inventoryList, procurementList, productionList, sales, customer, providerList = env.create_env(
        I, P)

    # 코드에 들어가는 옵션값
    total_cost_per_day = []
    # Run the simulation
    state = np.array([inven.level for inven in inventoryList]
                     )  # Get the inventory levels
    state_size = len(inventoryList)  # Number of inventories
    state = state.reshape(1, state_size)
    agent = DQNAgent(state_size, action_space, discount_factor,
                     epsilon_greedy, epsilon_min, epsilon_decay,
                     learning_rate, max_memory_size, target_update_frequency)
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

                    env.cal_cost(inventoryList, procurementList,
                                 productionList, sales, total_cost_per_day)

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

                        simpy_env, inventoryList, procurementList, productionList, sales, customer, providerList = env.create_env(
                            I, P)

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
