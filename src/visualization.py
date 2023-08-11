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
