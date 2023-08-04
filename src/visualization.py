import matplotlib.pyplot as plt
import seaborn as sns


'''
#visualization
class visualization:
    def __init__(self, inventory):
        self.inventory=inventory
        self.cost_list=[]    
        self.level_list=[]

    def return_list(self):
       self.level_list.append(self.inventory.level_over_time)
       self.cost_list.append(self.inventory.inventory_cost_over_time)
       return self.level_list,self.cost_list   #main문으로 데이터 반환 
        
        
    def inventory_level(self,level,item_name_list):
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 6))
        for i in range(len(level)):#item_id
            plt.plot(level[i][0],label=item_name_list[i])
        plt.xlabel('time[hours]')
        plt.ylabel('inventory')    
        plt.legend() 
        plt.grid(True)
       # plt.show()
       
    def inventory_cost(self,cost,item_name_list):
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 6))
        for i in range(len(cost)):#item_id
            plt.plot(cost[i][0],label=item_name_list[i])
        plt.xlabel('time[hours]')
        plt.ylabel('inventory')    
        plt.legend() 
        plt.grid(True)
        #plt.show()
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
       cal_cost = 
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
    
    
   