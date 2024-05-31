from InventoryMgtEnv import GymInterface
from config_SimPy import *
from config_RL import *
import numpy as np
import HyperparamTuning as ht  # Module for hyperparameter tuning
import time
from stable_baselines3 import DQN, DDPG, PPO
import visualization
from log_SimPy import *
from log_RL import *
import pandas as pd
import matplotlib.pyplot as plt
# Create environment
env = GymInterface()

# Function to evaluate the trained model


def evaluate_model(model, env, num_episodes):
    all_rewards = []  # List to store total rewards for each episode
    #XAI = []  # List for storing data for explainable AI purposes
    STATE_ACTION_REPORT_CORRECTION.clear()
    STATE_ACTION_REPORT_REAL.clear()
    ORDER_HISTORY=[]
    #For validation and visualization
    order_qty=[]
    demand_qty=[]
    onhand_inventory=[]
    test_order_mean = []  # List to store average orders per episode
    for i in range(num_episodes):
        ORDER_HISTORY.clear()
        episode_inventory=[[] for _ in range(len(I))]
        DAILY_REPORTS.clear()  # Clear daily reports at the start of each episode
        obs = env.reset()  # Reset the environment to get initial observation
        episode_reward = 0  # Initialize reward for the episode
        env.model_test=True
        done = False  # Flag to check if episode is finished
        while not done:
            
            for x in range(len(env.inventoryList)):
                episode_inventory[x].append(env.inventoryList[x].on_hand_inventory)
            action, _ = model.predict(obs)  # Get action from model
            # Execute action in environment
            obs, reward, done, _ = env.step(action)
            episode_reward += reward  # Accumulate rewards            
            
            ORDER_HISTORY.append(action[0])  # Log order history
            order_qty.append(action[-1])
            demand_qty.append(I[0]["DEMAND_QUANTITY"])
            
            
        onhand_inventory.append(episode_inventory)
        all_rewards.append(episode_reward)  # Store total reward for episode

        
        # Function to visualize the environment
        
  
        # Calculate mean order for the episode
        test_order_mean.append(sum(ORDER_HISTORY) / len(ORDER_HISTORY))
        COST_RATIO_HISTORY.append(env.cost_ratio)
    if VISUALIAZTION.count(1) > 0:
            visualization.visualization(DAILY_REPORTS)
    Visualize_invens(onhand_inventory,demand_qty,order_qty)
    cal_cost_avg()
    #print("Order_Average:", test_order_mean)
    '''
    if XAI_TRAIN_EXPORT:
        df = pd.DataFrame(XAI)  # Create a DataFrame from XAI data
        df.to_csv(f"{XAI_TRAIN}/XAI_DATA.csv")  # Save XAI data to CSV file
    '''
    if STATE_TEST_EXPORT:
        export_state("TEST")
    # Calculate mean reward across all episodes
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)  # Calculate standard deviation of rewards

    return mean_reward, std_reward  # Return mean and std of rewards
def Visualize_invens(inventory,demand_qty,order_qty):
    avg_inven=[[0 for _ in range(SIM_TIME)] for _ in range(len(I))]
    if VIZ_INVEN_PIE:
        for x in range(N_EVAL_EPISODES):
            for y in range(len(I)):
                for z in range(SIM_TIME):
                    avg_inven[y][z]+=inventory[x][y][z]

   
    if VIZ_INVEN_PIE:
        plt.pie([sum(avg_inven[x])/N_EVAL_EPISODES for x in range(len(I))],explode= [0.2, 0.2], labels=["Product","Material"], autopct='%1.1f%%')
        plt.legend()
        plt.show()
  
    if VIZ_INVEN_LINE:
        plt.plot(inventory[-1][0],"g--",label="Product")
        plt.plot(inventory[-1][1],"b--",label="Material_1")
        plt.plot(demand_qty[-SIM_TIME:],"r--",label="Demand_QTY")
        plt.plot(order_qty[-SIM_TIME:],"y--",label="ORDER")
        plt.legend()
        plt.show()
    
def cal_cost_avg():
    #Temp_Dict
    cost_avg={
    'Holding cost': 0,
    'Process cost': 0,
    'Delivery cost': 0,
    'Order cost': 0,
    'Shortage cost': 0
}
    #Temp_List
    total_avg=[]

    #Cal_cost_AVG
    for x in range(N_EVAL_EPISODES):
        for key in COST_RATIO_HISTORY[x].keys():
            cost_avg[key]+=COST_RATIO_HISTORY[x][key]
        total_avg.append(sum(COST_RATIO_HISTORY[x].values()))
    for key in cost_avg.keys():
        cost_avg[key]=cost_avg[key]/N_EVAL_EPISODES
    #Visualize
    if VIZ_COST_PIE:
        fig, ax = plt.subplots()
        plt.pie(cost_avg.values(), explode= [0.2, 0.2, 0.2, 0.2,0.2], labels=cost_avg.keys(), autopct='%1.1f%%')
        plt.show()
    if VIZ_COST_BOX:
        plt.boxplot(total_avg)
        plt.show()

def export_state(Record_Type):
    state_corr=pd.DataFrame(STATE_ACTION_REPORT_CORRECTION)
    state_real=pd.DataFrame(STATE_ACTION_REPORT_REAL)
    if Record_Type=='TEST':
        state_corr.dropna(axis=0,inplace=True)
        state_real.dropna(axis=0,inplace=True)
    print(state_real)
    columns_list=['Prod. InvenLevel', 'Prod. DailyChange', 'Mat. InvenLevel', 'Mat. DailyChange',
                'Remaining Demand','ACTION']
    '''
    for keys in I:
        columns_list.append(f"{I[keys]['NAME']}'s inventory")
        columns_list.append(f"{I[keys]['NAME']}'s Change")
    
    columns_list.append("Remaining Demand")
    columns_list.append("Action")
    '''
    state_corr.columns=columns_list
    state_real.columns=columns_list
    state_corr.to_csv(f'{STATE}/STATE_ACTION_REPORT_CORRECTION_{Record_Type}.csv')
    state_real.to_csv(f'{STATE}/STATE_ACTION_REPORT_REAL_{Record_Type}.csv')

# Function to build the model based on the specified reinforcement learning algorithm


def build_model():
    if RL_ALGORITHM == "DQN":
        model = DQN("MlpPolicy", env, verbose=0)
        # model = DQN("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
        #             batch_size=BEST_PARAMS['batch_size'], verbose=0)
    elif RL_ALGORITHM == "DDPG":
        model = DQN("MlpPolicy", env, verbose=0)
        # model = DDPG("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
        #              batch_size=BEST_PARAMS['batch_size'], verbose=0)
    elif RL_ALGORITHM == "PPO":
        model = PPO("MlpPolicy", env, verbose=0,n_steps=SIM_TIME)
        # model = PPO("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
        #             batch_size=BEST_PARAMS['batch_size'], verbose=0)
        print(env.observation_space)
    return model


'''
def export_report(inventoryList):
    for x in range(len(inventoryList)):
        for report in DAILY_REPORTS:
            export_Daily_Report.append(report[x])
    daily_reports = pd.DataFrame(export_Daily_Report)
    daily_reports.columns = ["Day", "Name", "Type",
                         "Start", "Income", "Outcome", "End"]
    daily_reports.to_csv("./Daily_Report.csv")
'''


# Start timing the computation
start_time = time.time()

# Run hyperparameter optimization if enabled
if OPTIMIZE_HYPERPARAMETERS:
    ht.run_optuna(env)

# Build the model
model = build_model()
# Train the model
model.learn(total_timesteps=SIM_TIME * N_EPISODES)
model.save("PPO_MODEL")
training_end_time=time.time()
if STATE_TRAIN_EXPORT:
    export_state('TRAIN')
    

# Optionally render the environment
env.render()

# Evaluate the trained model
mean_reward, std_reward = evaluate_model(model, env, N_EVAL_EPISODES)
print(
    f"Mean reward over {N_EVAL_EPISODES} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")


# Calculate computation time and print it
end_time = time.time()
print(f"Computation time: {(end_time - start_time)/60:.2f} minutes \n",
      f"Training time: {(training_end_time - start_time)/60:.2f} minutes \n ",
      f"Test time:{(end_time - training_end_time)/60:.2f} minutes")
