import matplotlib.pyplot as plt
from config_SimPy import *
from config_RL import *


def visualization(export_Daily_Report):
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
            # Record Onhand inventory at day end
            temp.append(export_Daily_Report[x][id*7+6])
        Visual_Dict[export_Daily_Report[0][id*8+2]].append(temp)  # Update
        Visual_Dict['Keys'][export_Daily_Report[0][2+id*8]
                            ].append(export_Daily_Report[0][id * 8+1])  # Update Keys
    # Number of output inventory types
    visual = VISUALIAZTION.count(1)
    # Count to specify the key
    count_type = 0
    # Variable for specifying the position of the graph
    graph_place = 1
    for x in VISUALIAZTION:
        # count for searching inventory of that type
        count = 0
        if x == 1:
            plt.subplot(int(f"{visual}1{graph_place}"))
            graph_place += 1
            for lst in Visual_Dict[Key[count_type]]:
                plt.plot(
                    lst, label=Visual_Dict['Keys'][Key[count_type]][count])
                plt.legend()
                count += 1
        count_type += 1
    path = os.path.join(GRAPH_FOLDER, f'그래프.png')
    plt.savefig(path)
    plt.clf()
