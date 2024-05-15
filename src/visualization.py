import matplotlib.pyplot as plt
from config_SimPy import *
from config_RL import *

def visualization(export_Daily_Report, i):
    Visual_Dict = {
        'Material': [],
        'WIP': [],
        'Product': [],
        'Keys': {'Material': [], 'WIP': [], 'Product': []}
    }
    Key = ['Material', 'WIP', 'Product']

    for id in I.keys():
        temp = []
        for x in range(SIM_TIME):
            temp.append(export_Daily_Report[x][id*7+6])#Record Onhand inventory at day end
        Visual_Dict[export_Daily_Report[0][id*7+2]].append(temp)#Update 
        Visual_Dict['Keys'][export_Daily_Report[0][2+id*7]].append(export_Daily_Report[0][id *7+1])#Update Keys
    visual = VISUALIAZTION.count(1)
    count_type = 0
    cont_len = 1
    for x in VISUALIAZTION:
        cont = 0
        if x == 1:
            plt.subplot(int(f"{visual}1{cont_len}"))
            cont_len += 1
            for lst in Visual_Dict[Key[count_type]]:
                plt.plot(lst, label=Visual_Dict['Keys'][Key[count_type]][cont])
                plt.legend()
                cont += 1
        count_type += 1
    path = os.path.join(GRAPH_FOLDER, f'그래프{i}.png')
    plt.savefig(path)
    plt.clf()
