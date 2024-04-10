from sklearn.tree import DecisionTreeClassifier, export_text  #sklearn에서 의사결정트리 모델을 텍스트로 내보냄 
import numpy as np
import pandas as pd
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

df=pd.read_csv('./XAI_DATA.csv')
# 가상의 학습 데이터 생성
X = df.iloc[:,1:-1]
y = df.iloc[:,-1:]
print(X,y)
# 의사 결정 트리 학습
clf = DecisionTreeClassifier()
print('start_fit')
clf = clf.fit(X, y)

# XAI 생성
# DOT 형식 데이터를 시각화하여 그래프 생성
# 의사 결정 트리를 시각화할 DOT 형식의 데이터 생성
#dot_data = export_graphviz(clf, out_file=None, 
#                           feature_names=['Product', 'R1','Demand'],  
#                           class_names=['Action0', 'Action1','Action2', 'Action3','Action4', 'Action5'],  
#                           filled=True, rounded=True,  
#                           special_characters=True)
#graph = graphviz.Source(dot_data)  
 
# 그래프를 PDF 파일로 저장하거나, 화면에 표시할 수 있습니다.
#graph.render("decision_tree_visualization")
#graph.view()
plt.figure(figsize=(20,10))
a = plot_tree(clf, out_file=None, 
                feature_names=['Product', 'R1','Demand'],  
                        class_names=['Action0', 'Action1','Action2', 'Action3','Action4', 'Action5'],  
                        filled=True, rounded=True,  
                        special_characters=True)

plt.show()