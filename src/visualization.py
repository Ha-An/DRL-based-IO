import matplotlib.pyplot as plt
import seaborn as sns

'''
class visualization:
    def __init__(self, inventory, item_name):
        self.inventory = inventory
        self.item_name = item_name
        
    

    def inventory_level_graph(self):
        cont=0
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 6))
        for inven in self.inventory:
            
            plt.plot(self.inventory.level_over_time, label=(' %d : inventory_level',cont))
            cont=+1
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
        plt.show()
       
    def inventory_cost(self,cost,item_name_list):
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 6))
        for i in range(len(cost)):#item_id
            plt.plot(cost[i][0],label=item_name_list[i])
        plt.xlabel('time[hours]')
        plt.ylabel('inventory')    
        plt.legend() 
        plt.grid(True)
        plt.show()
    
    
   