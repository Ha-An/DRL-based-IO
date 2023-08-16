import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class visualization:
    def __init__(self, inventory, item_name):
        self.inventory = inventory
        self.item_name = item_name

    def inventory_level_graph(self):
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 6))
        plt.plot(self.inventory.level_over_time, label='inventory_level')
        plt.xlabel('time[days]')
        plt.ylabel('inventory')
        plt.title(f'{self.item_name} inventory_level')
        plt.legend()
        plt.grid(True)
        plt.show()

    def inventory_cost_graph(self):
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 6))
        plt.plot(self.inventory.cost_over_time, label='inventory_cost')
        plt.xlabel('time[days]')
        plt.ylabel('inventory_cost')
        plt.title(f'{self.item_name} inventory_cost')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_learning_history(history):
        fig = plt.figure(1, figsize=(14, 5))
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(history, lw=4,
                 marker='o', markersize=10)
        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.xlabel('Episodes', size=20)
        plt.ylabel('# Total Rewards', size=20)
        plt.show()


'''
import matplotlib.pyplot as plt
import seaborn as sns
class visualization:
    def __init__(self, inventory,cal_cost):
        self.inventory=inventory
        self.cost_list=[]    
        self.level_list=[]
        self.cal_cost = cal_cost
    def return_list(self):
       self.level_list.append(self.inventory.level_over_time)
       self.cost_list.append(self.inventory.inventory_cost_over_time)
       #cal_cost = 
       return self.level_list,self.cost_list 
    def plot_inventory_graphs(self, level, cost, item_name_list):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        
        # 첫 번째 그래프 (inventory_level)를 왼쪽
        sns.set(style="darkgrid")
        for i in range(len(level)):
            axes[0].plot(level[i][0], label=item_name_list[i])
        axes[0].set_xlabel('hours')
        axes[0].set_ylabel('level')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_title("inventory level")
        
        # 두 번째 그래프 (inventory_cost)를 오른쪽
        sns.set(style="darkgrid")
        for i in range(len(cost)):
            axes[1].plot(cost[i][0], label=item_name_list[i])
        axes[1].set_xlabel('hours')
        axes[1].set_ylabel('cost')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_title("inventory cost")
        
        plt.tight_layout()
        plt.show()
        
        # 세 번째 그래프 (total_cost)
        sns.set(style="darkgrid")
        for i in range(len(cost)):
            axes[2].plot(cost[i][0], label=item_name_list[i])
        axes[2].set_xlabel('hours')
        axes[2].set_ylabel('cost')
        axes[2].legend()
        axes[2].grid(True)
        axes[2].set_title("total cost")
        
        plt.tight_layout()
        plt.show()

    def plot_learning_history(history):
        fig = plt.figure(1, figsize=(14, 5))
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(history, lw=4,
                 marker='o', markersize=10)
        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.xlabel('Episodes', size=20)
        plt.ylabel('# Total Rewards', size=20)
        plt.show()
'''
