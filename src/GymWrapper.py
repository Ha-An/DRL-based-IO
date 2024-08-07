import gym
from gym import spaces
import numpy as np
from config_SimPy import *
from config_RL import *
import environment as env
from log_SimPy import *
from log_RL import *
import pandas as pd
import matplotlib.pyplot as plt
import visualization
from torch.utils.tensorboard import SummaryWriter


class GymInterface(gym.Env):
    def __init__(self):
        self.outer_end = False
        self.writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)
        print("Tensorboard Directory: :", TENSORFLOW_LOGS)
        super(GymInterface, self).__init__()

        # Scenario initialization for the demand
        if DEMAND_DIST_TYPE == "UNIFORM":
            demand_dist = {"Dist_Type": "UNIFORM",
                           "min": 8, "max": 13}  # Default scenario
            # demand_dist = {"Dist_Type": "UNIFORM",
            #                  "min": 5, "max": 12}
            # demand_dist = {"Dist_Type": "UNIFORM",
            #                  "min": 11, "max": 18}
        elif DEMAND_DIST_TYPE == "GAUSSIAN":
            leadtime_dist = {"Dist_Type": "GAUSSIAN",
                             "mean": 11, "std": 4}
        # Scenario initialization for the demand
        if LEAD_DIST_TYPE == "UNIFORM":
            leadtime_dist = {"Dist_Type": "UNIFORM",
                             "min": 1, "max": 1}  # Default scenario
        elif LEAD_DIST_TYPE == "GAUSSIAN":
            leadtime_dist = {"Dist_Type": "GAUSSIAN",
                             "mean": 3, "std": 3}
        self.scenario = {"DEMAND": demand_dist, "LEADTIME": leadtime_dist}

        self.shortages = 0
        self.total_reward_over_episode = []
        self.total_reward = 0
        self.cur_episode = 1  # Current episode
        self.cur_outer_loop = 1  # Current outer loop
        self.cur_inner_loop = 1  # Current inner loop
        self.scenario_batch_size = 99999  # Initialize the scenario batch size

        # For functions that only work when testing the model
        self.model_test = False
        # Record the cumulative value of each cost
        self.cost_dict = {
            'Holding cost': 0,
            'Process cost': 0,
            'Delivery cost': 0,
            'Order cost': 0,
            'Shortage cost': 0
        }
        os = []

        # Action space, observation space
        if RL_ALGORITHM == "PPO":
            # Define action space
            actionSpace = []
            for i in range(len(I)):
                if I[i]["TYPE"] == "Material":
                    actionSpace.append(len(ACTION_SPACE))
            self.action_space = spaces.MultiDiscrete(actionSpace)
            # if self.scenario["Dist_Type"] == "UNIFORM":
            #    k = INVEN_LEVEL_MAX*2+(self.scenario["max"]+1)

            # DAILY_CHANGE + INTRANSIT + REMAINING_DEMAND
            if USE_CORRECTION:
                os = [102 for _ in range(
                    len(I)*(1+DAILY_CHANGE)+MAT_COUNT*INTRANSIT+1)]
            else:
                os = [
                    INVEN_LEVEL_MAX * 2 + 1 for _ in range(len(I)*(1+DAILY_CHANGE)+MAT_COUNT*INTRANSIT+1)]
            '''
            - Inventory Level of Product
            - Daily Change of Product
            - Inventory Level of WIP
            - Daily Change of WIP
            - Inventory Level of Material
            - Daily Change of Material
            - Demand - Inventory Level of Product
            '''
            self.observation_space = spaces.MultiDiscrete(os)
        elif RL_ALGORITHM == "DQN":
            pass
        elif RL_ALGORITHM == "DDPG":
            pass
        print(os)

    def reset(self):
        # Initialize the total reward for the episode
        self.cost_dict = {
            'Holding cost': 0,
            'Process cost': 0,
            'Delivery cost': 0,
            'Order cost': 0,
            'Shortage cost': 0
        }
        # Initialize the simulation environment
        self.simpy_env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.providerList, self.daily_events = env.create_env(
            I, P, DAILY_EVENTS)
        env.simpy_event_processes(self.simpy_env, self.inventoryList, self.procurementList,
                                  self.productionList, self.sales, self.customer, self.providerList, self.daily_events, I, self.scenario)
        env.update_daily_report(self.inventoryList)

        state_real = self.get_current_state()
        state_correction = self.correct_state_for_SB3()
        if USE_CORRECTION:
            return state_correction
        else:
            return state_real

    def step(self, action):
        # Update the action of the agent
        if RL_ALGORITHM == "PPO":
            i = 0
            for _ in range(len(I)):
                if I[_]["TYPE"] == "Material":
                    # Set action as predicted value
                    I[_]["LOT_SIZE_ORDER"] = action[i]
                    # Set action to fixed value
                    # I[_]["LOT_SIZE_ORDER"] = ORDER_QTY[i]
                    i += 1
        elif RL_ALGORITHM == "DQN":
            pass

        # Capture the current state of the environment
        # current_state = env.cap_current_state(self.inventoryList)
        # Run the simulation for 24 hours (until the next day)
        # Action append
        STATE_ACTION_REPORT_REAL[-1].append(action)
        STATE_ACTION_REPORT_CORRECTION[-1].append(action)

        self.simpy_env.run(until=self.simpy_env.now + 24)
        env.update_daily_report(self.inventoryList)
        # Capture the next state of the environment
        state_real = self.get_current_state()
        state_corr = self.correct_state_for_SB3()
        # Set the next state
        if USE_CORRECTION:
            next_state = state_corr
        else:
            next_state = state_real
        # Calculate the total cost of the day
        env.Cost.update_cost_log(self.inventoryList)
        if PRINT_SIM:
            cost = dict(DAILY_COST_REPORT)
        # Cost Dict update
        for key in DAILY_COST_REPORT.keys():
            self.cost_dict[key] += DAILY_COST_REPORT[key]

        env.Cost.clear_cost()

        reward = -COST_LOG[-1]
        REWARD_LOG.append(COST_LOG[-1])
        self.total_reward += reward
        self.shortages += self.sales.num_shortages
        self.sales.num_shortages = 0

        if PRINT_SIM:
            # Print the simulation log every 24 hours (1 day)
            print(f"\nDay {(self.simpy_env.now+1) // 24}:")
            if RL_ALGORITHM == "PPO":
                i = 0
                for _ in range(len(I)):
                    if I[_]["TYPE"] == "Raw Material":
                        print(
                            f"[Order Quantity for {I[_]['NAME']}] ", action[i])
                        i += 1
            # SimPy simulation print
            for log in self.daily_events:
                print(log)
            print("[Daily Total Cost] ", -reward)
            for _ in cost.keys():
                print(_, cost[_])
            print("Total cost: ", -self.total_reward)
            if USE_CORRECTION:
                print("[CORRECTED_STATE for the next round] ",
                      [item for item in next_state])
            else:
                print("[REAL_STATE for the next round] ",  [
                      item-INVEN_LEVEL_MAX for item in next_state])

        self.daily_events.clear()

        # Check if the simulation is done
        done = self.simpy_env.now >= SIM_TIME * 24  # 예: SIM_TIME일 이후에 종료
        if done == True:
            if DRL_TENSORBOARD:
                self.writer.add_scalar(
                    "reward", self.total_reward, global_step=self.cur_episode)
                # Log each cost ratio at the end of the episode
                for cost_name, cost_value in self.cost_dict.items():
                    self.writer.add_scalar(
                        cost_name, cost_value, global_step=self.cur_episode)
                self.writer.add_scalars(
                    'Cost', self.cost_dict, global_step=self.cur_episode)
                print("Episode: ", self.cur_episode,
                      " / Total reward: ", self.total_reward)

            if self.outer_end == True and self.scenario_batch_size == self.cur_inner_loop:
                self.writer.add_scalar(
                    "inner_end/reward", self.total_reward, global_step=self.cur_episode)
            self.total_reward_over_episode.append(self.total_reward)
            self.total_reward = 0
            self.cur_episode += 1

        info = {}  # 추가 정보 (필요에 따라 사용)

        return next_state, reward, done, info

    def get_current_state(self):
        # Make State for RL
        state = []
        # Update STATE_ACTION_REPORT_REAL
        for id in range(len(I)):
            # ID means Item_ID, 7 means to the length of the report for one item
            # append On_Hand_inventory
            state.append(
                STATE_DICT[-1][f"On_Hand_{I[id]['NAME']}"]+INVEN_LEVEL_MAX)
            # append changes in inventory
            if DAILY_CHANGE == 1:
                # append changes in inventory
                state.append(
                    STATE_DICT[-1][f"Daily_Change_{I[id]['NAME']}"]+INVEN_LEVEL_MAX)
            if INTRANSIT == 1:
                if I[id]["TYPE"] == "Material":
                    # append Intransition inventory
                    state.append(
                        STATE_DICT[-1][f"In_Transit_{I[id]['NAME']}"])

        # Append remaining demand
        state.append(I[0]["DEMAND_QUANTITY"] -
                     self.inventoryList[0].on_hand_inventory+INVEN_LEVEL_MAX)
        STATE_ACTION_REPORT_REAL.append(
            [Item - INVEN_LEVEL_MAX for Item in state])
        return state

    # Min-Max Normalization
    def correct_state_for_SB3(self):
        # Find minimum Delta
        product_outgoing_correction = 0
        for key in P:
            # product_outgoing_correction = max(P[key]["PRODUCTION_RATE"] * max(P[key]['QNTY_FOR_INPUT_ITEM']), self.scenario["max"])
            product_outgoing_correction = max(
                P[key]["PRODUCTION_RATE"] * max(P[key]['QNTY_FOR_INPUT_ITEM']), INVEN_LEVEL_MAX)
        # Update STATE_ACTION_REPORT_CORRECTION.append(state_corrected)
        state_corrected = []
        for id in range(len(I)):
            # normalization Onhand inventory
            state_corrected.append(
                round((STATE_DICT[-1][f"On_Hand_{I[id]['NAME']}"]/INVEN_LEVEL_MAX)*100))
            if DAILY_CHANGE == 1:
                state_corrected.append(round(((STATE_DICT[-1][f"Daily_Change_{I[id]['NAME']}"]-(-product_outgoing_correction))/(
                    ACTION_SPACE[-1]-(-product_outgoing_correction)))*100))  # normalization changes in inventory
            if I[id]['TYPE'] == "Material":
                if INTRANSIT == 1:
                    state_corrected.append(round(
                        (STATE_DICT[-1][f"In_Transit_{I[id]['NAME']}"]-ACTION_SPACE[0])/(ACTION_SPACE[-1]*7-ACTION_SPACE[0])))

        # normalization remaining demand
        state_corrected.append(round(
            ((I[0]["DEMAND_QUANTITY"]-self.inventoryList[0].on_hand_inventory+INVEN_LEVEL_MAX)/(I[0]['DEMAND_QUANTITY']+INVEN_LEVEL_MAX))*100))
        STATE_ACTION_REPORT_CORRECTION.append(state_corrected)
        return STATE_ACTION_REPORT_CORRECTION[-1]

    def render(self, mode='human'):
        pass

    def close(self):
        # 필요한 경우, 여기서 리소스를 정리
        pass


#Function to evaluate the trained model
# def evaluate_model(model, env, num_episodes):
#     all_rewards = []  # List to store total rewards for each episode
#     # XAI = []  # List for storing data for explainable AI purposes
#     STATE_ACTION_REPORT_CORRECTION.clear()
#     STATE_ACTION_REPORT_REAL.clear()
#     ORDER_HISTORY = []
#     # For validation and visualization
#     Mat_Order = {}
#     for mat in range(MAT_COUNT):
#         Mat_Order[f"mat {mat}"] = []
#     demand_qty = []
#     onhand_inventory = []
#     test_order_mean = []  # List to store average orders per episode
#     for i in range(num_episodes):
#         ORDER_HISTORY.clear()
#         episode_inventory = [[] for _ in range(len(I))]
#         DAILY_REPORTS.clear()  # Clear daily reports at the start of each episode
#         obs = env.reset()  # Reset the environment to get initial observation
#         episode_reward = 0  # Initialize reward for the episode
#         env.model_test = True
#         done = False  # Flag to check if episode is finished
#         day = 1  # 차후 validaition끝나면 지울것
#         while not done:
#             for x in range(len(env.inventoryList)):
#                 episode_inventory[x].append(
#                     env.inventoryList[x].on_hand_inventory)
#             action, _ = model.predict(obs)  # Get action from model
#             # validation 끝나면 지울것
#             if VALIDATION:
#                 action = validation_input(day)
#             # Execute action in environment
#             if len(ORDER_QTY) != 0:
#                 action = ORDER_QTY
#             obs, reward, done, _ = env.step(action)
#             episode_reward += reward  # Accumulate rewards
#             ORDER_HISTORY.append(action[0])  # Log order history
#             for x in range(len(action)):
#                 #Mat_Order[f"mat {x}"].append(action[x])
#                 Mat_Order.append(I[1]["LOT_SIZE_ORDER"])
#             demand_qty.append(I[0]["DEMAND_QUANTITY"])
#             day += 1  # 추후 validation 끝나면 지울 것

#         onhand_inventory.append(episode_inventory)
#         all_rewards.append(episode_reward)  # Store total reward for episode
#         # Function to visualize the environment

#         # Calculate mean order for the episode
#         order_mean = []
#         for key in Mat_Order.keys():
#             order_mean.append(sum(Mat_Order[key]) / len(Mat_Order[key]))
#         test_order_mean.append(order_mean)
#         COST_HISTORY.append(env.cost_dict)
#     if VISUALIAZTION.count(1) > 0:
#         visualization.visualization(DAILY_REPORTS)
#     Visualize_invens(onhand_inventory, demand_qty, Mat_Order, all_rewards)
#     cal_cost_avg()
#     # print("Order_Average:", test_order_mean)
#     if STATE_TEST_EXPORT:
#         export_state("TEST")
#     # Calculate mean reward across all episodes
#     mean_reward = np.mean(all_rewards)
#     std_reward = np.std(all_rewards)  # Calculate standard deviation of rewards
#     return mean_reward, std_reward  # Return mean and std of rewards

def evaluate_model(model, env, num_episodes):
    all_rewards = []  # List to store total rewards for each episode
    # XAI = []  # List for storing data for explainable AI purposes
    STATE_ACTION_REPORT_CORRECTION.clear()
    STATE_ACTION_REPORT_REAL.clear()
    ORDER_HISTORY = []
    # For validation and visualization
    order_qty = []
    demand_qty = []
    onhand_inventory = []
    test_order_mean = []  # List to store average orders per episode
    for i in range(num_episodes):
        ORDER_HISTORY.clear()
        episode_inventory = [[] for _ in range(len(I))]
        DAILY_REPORTS.clear()  # Clear daily reports at the start of each episode
        obs = env.reset()  # Reset the environment to get initial observation
        episode_reward = 0  # Initialize reward for the episode
        env.model_test = True
        done = False  # Flag to check if episode is finished
        day = 1  # 차후 validaition끝나면 지울것
        while not done:

            for x in range(len(env.inventoryList)):
                episode_inventory[x].append(
                    env.inventoryList[x].on_hand_inventory)
            action, _ = model.predict(obs)  # Get action from model
            # validation 끝나면 지울것
            if VALIDATION:
                action = validation_input(day)
            # Execute action in environment => 현재 Material 1개에 대한 action만 코딩되어 있음. 추후 여러 Material에 대한 action을 코딩해야 함.
            obs, reward, done, _ = env.step(action)
            episode_reward += reward  # Accumulate rewards
            ORDER_HISTORY.append(action[0])  # Log order history
            # ORDER_HISTORY.append(I[1]["LOT_SIZE_ORDER"])  # Log order history
            order_qty.append(action[-1])
            # order_qty.append(I[1]["LOT_SIZE_ORDER"])
            demand_qty.append(I[0]["DEMAND_QUANTITY"])
            day += 1  # 추후 validation 끝나면 지울 것

        onhand_inventory.append(episode_inventory)
        all_rewards.append(episode_reward)  # Store total reward for episode
        # Function to visualize the environment

        # Calculate mean order for the episode
        test_order_mean.append(sum(ORDER_HISTORY) / len(ORDER_HISTORY))
        COST_HISTORY.append(env.cost_dict)
    if VISUALIAZTION.count(1) > 0:
        visualization.visualization(DAILY_REPORTS)
    Visualize_invens(onhand_inventory, demand_qty, order_qty, all_rewards)
    cal_cost_avg()
    # print("Order_Average:", test_order_mean)
    if STATE_TEST_EXPORT:
        export_state("TEST")
    # Calculate mean reward across all episodes
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)  # Calculate standard deviation of rewards
    return mean_reward, std_reward  # Return mean and std of rewards


def cal_cost_avg():
    # Temp_Dict
    cost_avg = {
        'Holding cost': 0,
        'Process cost': 0,
        'Delivery cost': 0,
        'Order cost': 0,
        'Shortage cost': 0
    }
    # Temp_List
    total_avg = []

    # Cal_cost_AVG
    for x in range(N_EVAL_EPISODES):
        for key in COST_HISTORY[x].keys():
            cost_avg[key] += COST_HISTORY[x][key]
        total_avg.append(sum(COST_HISTORY[x].values()))
    for key in cost_avg.keys():
        cost_avg[key] = cost_avg[key]/N_EVAL_EPISODES
    # Visualize
    if VIZ_COST_PIE:
        plt.figure(figsize=(20, 20))
        plt.pie(cost_avg.values(), explode=[
                0.2, 0.2, 0.2, 0.2, 0.2], labels=cost_avg.keys(), autopct='%1.1f%%')
        plt.show()
    if VIZ_COST_BOX:
        plt.boxplot(total_avg)
        plt.show()


def Visualize_invens(inventory, demand_qty, Mat_Order, all_rewards):
    best_reward = -99999999999999
    best_index = 0
    for x in range(N_EVAL_EPISODES):
        if all_rewards[x] > best_reward:
            best_reward = all_rewards[x]
            best_index = x

    avg_inven = [[0 for _ in range(SIM_TIME)] for _ in range(len(I))]
    lable = []
    for id in I.keys():
        lable.append(I[id]["NAME"])

    if VIZ_INVEN_PIE:
        plt.figure(figsize=(20, 20))
        for x in range(N_EVAL_EPISODES):
            for y in range(len(I)):
                for z in range(SIM_TIME):
                    avg_inven[y][z] += inventory[x][y][z]

        plt.pie([sum(avg_inven[x])/N_EVAL_EPISODES for x in range(len(I))],
                explode=[0.2 for _ in range(len(I))], labels=lable, autopct='%1.1f%%')
        plt.legend()
        plt.show()

    if VIZ_INVEN_LINE:
        plt.figure(figsize=(20, 20))
        # Inven Line
        for id in I.keys():
            # Visualize the inventory levels of the best episode
            plt.plot(inventory[best_index][id], label=lable[id])
        plt.plot(demand_qty[-SIM_TIME:], "y--", label="Demand_QTY")
        # Order_Line
        for key in Mat_Order.keys():
            plt.plot(Mat_Order[key][-SIM_TIME:], label=f"ORDER {key}")
        plt.legend()
        plt.show()


def export_state(Record_Type):
    state_corr = pd.DataFrame(STATE_ACTION_REPORT_CORRECTION)
    state_real = pd.DataFrame(STATE_ACTION_REPORT_REAL)

    if Record_Type == 'TEST':
        state_corr.dropna(axis=0, inplace=True)
        state_real.dropna(axis=0, inplace=True)

    columns_list = []
    for id in I.keys():
        if I[id]["TYPE"] == 'Material':
            columns_list.append(f"{I[id]['NAME']}.InvenLevel")
            if DAILY_CHANGE:
                columns_list.append(f"{I[id]['NAME']}.DailyChange")
            if INTRANSIT:
                columns_list.append(f"{I[id]['NAME']}.Intransit")
        else:
            columns_list.append(f"{I[id]['NAME']}.InvenLevel")
            if DAILY_CHANGE:
                columns_list.append(f"{I[id]['NAME']}.DailyChange")
    columns_list.append("Remaining_Demand")
    columns_list.append("Action")
    '''
    for keys in I:
        columns_list.append(f"{I[keys]['NAME']}'s inventory")
        columns_list.append(f"{I[keys]['NAME']}'s Change")
    
    columns_list.append("Remaining Demand")
    columns_list.append("Action")
    '''
    state_corr.columns = columns_list
    state_real.columns = columns_list

    state_corr.to_csv(
        f'{STATE}/STATE_ACTION_REPORT_CORRECTION_{Record_Type}.csv')
    state_real.to_csv(f'{STATE}/STATE_ACTION_REPORT_REAL_{Record_Type}.csv')
