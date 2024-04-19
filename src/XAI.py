from sklearn.tree import DecisionTreeClassifier, export_text
import numpy as np
import pandas as pd
from sklearn.tree import export_graphviz
import graphviz
from config_RL import *

df=pd.read_csv('./XAI_DATA.csv')
# XAI data classification
X = df.iloc[:,1:-1]
y = df.iloc[:,-1:]
print(X,y)
# Decision tree learning
clf = DecisionTreeClassifier()
print('start_fit')
clf = clf.fit(X, y)

# XAI
# Visualize DOT format data to create graphs
# Generate data in DOT format to visualize decision trees
dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=FEATUER_NAME,  
                           class_names=[ACTION_SPACE],  
                           filled=True, rounded=True,  
                           special_characters=True)
graph = graphviz.Source(dot_data)  

 
#you can save the graph as a PDF file or display it on the screen.
#graph.render("decision_tree_visualization")
graph.view()