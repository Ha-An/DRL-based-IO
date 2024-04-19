import gym
from gym import spaces
import numpy as np
from config_SimPy import *
from config_RL import *
import environment as env
import matplotlib.pyplot as plt
import optuna.visualization as vis
from log import *
import random
from torch.utils.tensorboard import SummaryWriter


class GymInterface(gym.Env):
    def __init__(self):
        self.shortages = 0
        self.writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)
        os = []
        super(GymInterface, self).__init__()
        # Action space, observation space
        if RL_ALGORITHM == "DQN":
            # Define action space
            self.action_space = spaces.Discrete(len(ACTION_SPACE))
            # Define observation space:
            # self.observation_space = spaces.Box(low=0, high=INVEN_LEVEL_MAX, shape=(len(I),), dtype=int)
            os = []
            for _ in range(len(I)):
                os.append(INVEN_LEVEL_MAX+1)
            if STATE_DEMAND:
                os.append(DEMAND_QTY_MAX+1+DELTA_MIN)
                os.append(DEMAND_QTY_MAX+1)
            self.observation_space = spaces.MultiDiscrete(os)
        elif RL_ALGORITHM == "DDPG":
            # self.action_space = spaces.Box(low=0, high=len(ACTION_SPACE), shape=(1,), dtype=np.float32)
            actionSpace_low = []
            actionSpace_high = []
            for _ in range(len(I)):
                if I[_]["TYPE"] == "Material":
                    actionSpace_low.append(0)
                    actionSpace_high.append(len(ACTION_SPACE))
            self.action_space = spaces.Box(low=np.array(actionSpace_low), high=np.array(
                actionSpace_high), dtype=np.float32)
            # self.observation_space = spaces.Box(low=0, high=INVEN_LEVEL_MAX, shape=(len(I),), dtype=np.float32)
            os = [INVEN_LEVEL_MAX+1 for _ in range(len(I))]
            if STATE_DEMAND:
                os.append(DEMAND_QTY_MAX + 1)
                os.append(DEMAND_QTY_MAX+1)
            self.observation_space = spaces.Discrete(os)
        elif RL_ALGORITHM == "PPO":
            # Define action space
            actionSpace = []
            for _ in range(len(I)):
                if I[_]["TYPE"] == "Material":
                    actionSpace.append(len(ACTION_SPACE))
            self.action_space = spaces.MultiDiscrete(actionSpace)
            # Define observation space:
            for i in range(len(I)):
                os.append(INVEN_LEVEL_MAX+1)
                os.append(INVEN_LEVEL_MAX+1+DELTA_MIN)
            if STATE_DEMAND:
                os.append(DEMAND_QTY_MAX + 1+EXPECTED_PRODUCT_MAX)
            self.observation_space = spaces.MultiDiscrete(os)
            print(os)
        # print(self.observation_space)
        # Simpy environment
        # self.simpy_env = simpy.Environment()
        # self.simpy_env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.providerList, self.daily_events = env.create_env(
        #    I, P, DAILY_EVENTS)
        # print(self.simpy_env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.providerList, self.daily_events)
        self.total_reward_over_episode = []
        self.total_reward = 0
        self.num_episode = 1

    def reset(self):
        # Initialize the simulation environment
        print("\nEpisode: ", self.num_episode)
        self.simpy_env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.providerList, self.daily_events = env.create_env(
            I, P, DAILY_EVENTS)
        env.simpy_event_processes(self.simpy_env, self.inventoryList, self.procurementList,
                                  self.productionList, self.sales, self.customer, self.providerList, self.daily_events, I)
        self.shortages = 0
        return self.cap_current_state()

    def step(self, action):
        # Update the action of the agent
        if RL_ALGORITHM == "DQN":
            I[1]["LOT_SIZE_ORDER"] = action
            # I[0]["DEMAND_QUANTITY"] = random.randint(DEMAND_QTY_MIN, DEMAND_QTY_MAX)
            # I[0]["DUE_DATE"] = random.randint(DUE_DATE_MIN, DUE_DATE_MAX)

        elif RL_ALGORITHM == "DDPG":
            i = 0
            for _ in range(len(I)):
                if I[_]["TYPE"] == "Material":
                    # I[_]["LOT_SIZE_ORDER"] = int(round(action[i]))
                    I[_]["LOT_SIZE_ORDER"] = action[i]
                    i += 1
        elif RL_ALGORITHM == "PPO":
            i = 0
            for _ in range(len(I)):
                if I[_]["TYPE"] == "Material":
                    # I[_]["LOT_SIZE_ORDER"] = action[i]
                    I[_]["LOT_SIZE_ORDER"] = ORDER_QTY
                    i += 1

        # Capture the current state of the environment
        # current_state = env.cap_current_state(self.inventoryList)
        # Run the simulation for 24 hours (until the next day)
        env.update_daily_report(self.inventoryList)
        self.simpy_env.run(until=self.simpy_env.now + 24)
        # Capture the next state of the environment
        next_state = self.cap_current_state()
        # Calculate the total cost of the day
        env.Cost.update_cost_log(self.inventoryList)
        env.Cost.clear_cost()
        # reward = -COST_LOG[-1]-next_state[-1]*I[0]["SHORTAGE_COST_PRO"]
        reward = -COST_LOG[-1]
        self.total_reward += reward
        self.shortages += self.sales.num_shortages
        self.sales.num_shortages = 0
        '''
        s = []
        for _ in range(len(self.inventoryList)):
            s.append(self.inventoryList[_].current_level)
        daily_total_cost = env.cal_daily_cost_DESC(s[0], s[1], action)
        '''
        if PRINT_SIM_EVENTS:
            # Print the simulation log every 24 hours (1 day)
            print(f"\nDay {(self.simpy_env.now+1) // 24}:")
            if RL_ALGORITHM == "DQN":
                print(f"[Order Quantity for {I[1]['NAME']}] ", action)
            else:
                i = 0
                for _ in range(len(I)):
                    if I[_]["TYPE"] == "Raw Material":
                        print(
                            f"[Order Quantity for {I[_]['NAME']}] ", action[i])
                        i += 1
            for log in self.daily_events:
                print(log)
            print("[Daily Total Cost] ", -reward)
            print("[STATE for the next round] ", next_state)
        self.daily_events.clear()

        # Check if the simulation is done
        done = self.simpy_env.now >= SIM_TIME * 24  # 예: SIM_TIME일 이후에 종료
        if done == True:
            self.writer.add_scalar(
                "reward", self.total_reward, global_step=self.num_episode)
            self.writer.add_scalar(
                "shortage", self.shortages, global_step=self.num_episode)

            print("Total reward: ", self.total_reward)
            self.total_reward_over_episode.append(self.total_reward)
            self.total_reward = 0
            self.num_episode += 1

        info = {}  # 추가 정보 (필요에 따라 사용)

        # self.all_order_quantities.append(action)
        # self.all_rewards.append(reward)
        # self.all_inventory_levels.append((next_state[0], next_state[1]))
        return next_state, reward, done, info

    def cap_current_state(self):
        state = []
        for inven in self.inventoryList:

            # Function to capture the current state of the inventory
            state.append(
                inven.daily_inven_report[4]-inven.daily_inven_report[5]+DELTA_MIN)
            state.append(inven.daily_inven_report[6])
            # Reset Report
            inven.daily_inven_report = [f"Day {inven.env.now//24}", I[inven.item_id]['NAME'], I[inven.item_id]['TYPE'],
                                        inven.on_hand_inventory, 0, 0, 0]  # inventory report
        if STATE_DEMAND:
            state.append(I[0]['DEMAND_QUANTITY'] -
                         self.inventoryList[0].on_hand_inventory+EXPECTED_PRODUCT_MAX)

        return state

    def render(self, mode='human'):
        pass
        # if EPISODES == 1:
        #     self.visualize()
        # else:
        #     if OPTIMIZE_HYPERPARAMETERS:
        #         pass
        #     else:
        #         self.visualize()

        #     # Total rewards over episodes
        #     fig = plt.figure(1, figsize=(14, 5))
        #     ax = fig.add_subplot(1, 1, 1)
        #     plt.plot(self.total_reward_over_episode, lw=4,
        #              marker='o', markersize=10)
        #     ax.tick_params(axis='both', which='major', labelsize=15)
        #     plt.xlabel('Episodes', size=20)
        #     plt.ylabel('Total Rewards', size=20)
        #     plt.show()

    def close(self):
        # 필요한 경우, 여기서 리소스를 정리
        pass
