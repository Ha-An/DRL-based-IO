from config import *
import environment as env
import numpy as np
import random


def policy(V):
    for i in range(10):
        for j in range(8):
            a = [V[i][j], V[i][j+1], V[i][j+2]]
            print("policy", (i, j), np.argmax(a))


num_actions = 3
num_states = 100

V = np.zeros((10, 10))

gamma = 0.9
num_iterations = 100
daily_events = []
agent = 1
states = []
values = 10
for i in range(values):
    for j in range(values):
        states.append([i, j])
print(states)


simpy_env, inventoryList, procurementList, productionList, sales, customer, providerList, daily_events = env.create_env(
    I, P, daily_events)
env.simpy_event_processes(agent, simpy_env, inventoryList, procurementList,
                          productionList, sales, customer, providerList, daily_events, I)
new_V = np.zeros((10, 10))
for _ in range(num_iterations):
    for seq in range(len(states)):
        max_q_value = float("-inf")
        q_value = 0
        for a in range(num_actions):
            I[1]['LOT_SIZE_ORDER'] = a
            inventoryList[0].current_level = states[seq][0]
            inventoryList[1].current_level = states[seq][1]
            print(states[seq][0], states[seq][1])
            state = [states[seq][0], states[seq][1]]
            for i in range(24):
                # Run the simulation until the next hour
                simpy_env.run(until=simpy_env.now+1)

                if (i+1) % 24 == 0:  # Daily time step

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

            next_state = np.array(
                [inven.current_level for inven in inventoryList])
            print(next_state)
            print("daily", daily_total_cost)
            # if next_state[0] <= 9 & next_state[1] <= 9 & next_state[0] >= 0 & next_state[1] >= 0:
            q_value = (-daily_total_cost + gamma *
                       V[next_state[0]][next_state[1]])
            if q_value > max_q_value:
                max_q_value = q_value
            print(q_value)
            print(max_q_value)
            print("______________________")
        new_V[state[0]][state[1]] = max_q_value
        V = new_V

print("Optimal Value Function:")
print(V)
policy(V)
