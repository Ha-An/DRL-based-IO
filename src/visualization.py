import matplotlib.pyplot as plt
from log_SimPy import *
from config_SimPy import *


def viz_sq():
    for graph in GRAPH_LOG.keys():
        plt.plot(GRAPH_LOG[graph], label=f'{graph}')
    plt.legend()
    plt.show()


def record_graph(item):
    for info in item.keys():
        GRAPH_LOG[item[info]['NAME']] = []
        if item[info]['TYPE'] == 'Material':
            GRAPH_LOG[f"{item[info]['NAME']}_in_transition_inventory"] = []
    print(GRAPH_LOG)
