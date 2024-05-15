from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd
from sklearn.tree import export_graphviz
import graphviz
import os
from call_shap import cal_shap #Call cal_shap function

SHAP_PLOT_TYPE='bar' #auto, bar, violin, dot, layered_violin, heatmap, waterfall, image
'''
'auto': Automatically selects the appropriate plot type based on the Shap value.
'bar': Creates a bar graph that displays the average Shap value for each feature and the absolute average effect of that feature value.
'violin': Visualize the distribution of Shap values for each feature as a violin plot.
'dot': Creates a dot plot that displays the distribution of Shap values for each feature as dots.
'layered_violin': Visualizes the violin plot divided into multiple layers.
'heatmap': Creates a heatmap of Shap values to visualize interactions between features.
'waterfall': Visualize the contribution of Shap values for each feature as a stacked bar graph.
'image': Supports visualization of Shap values for image data
'''

def read_path():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    result_csv_folder = os.path.join(parent_dir, "result_CSV")
    STATE_folder = os.path.join(result_csv_folder, "state")
    Data_place=os.path.join(STATE_folder,f"Train_{len(os.listdir(STATE_folder))}")
    return Data_place

#Read newest Test Dataset
df = pd.read_csv(
    os.path.join(f"{read_path()}","STATE_ACTION_REPORT_REAL_TEST.csv"))

# XAI data classification
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1:]
# Decision tree learning
clf = DecisionTreeClassifier(
    criterion='gini',          # 'gini' 또는 'entropy'
    max_depth=6,               # 트리의 최대 깊이
    min_samples_split=20,      # 노드를 분할하기 위한 최소 샘플 수
    min_samples_leaf=10        # 잎 노드가 가지고 있어야 할 최소 샘플 수
)
print('start_fit')
clf = clf.fit(X, y)
# Extract Unique Actions of test datasetpython py
actions=df['ACTION'].unique()



def simplify_tree(tree, node_id=0):
    if tree.children_left[node_id] == tree.children_right[node_id] == -1:
        return tree.value[node_id].argmax(), True, int(tree.value[node_id].sum())

    left_class, left_is_leaf, left_samples = simplify_tree(tree, tree.children_left[node_id])
    right_class, right_is_leaf, right_samples = simplify_tree(tree, tree.children_right[node_id])

    if left_is_leaf and right_is_leaf and left_class == right_class:
        
        tree.children_left[node_id] = -1
        tree.children_right[node_id] = -1
        
        return left_class, True, left_samples + right_samples

    return tree.value[node_id].argmax(), False, int(tree.value[node_id].sum())

simplify_tree(clf.tree_)



FEATURE_NAME = df.columns[1:-1]
# XAI
# Visualize DOT format data to create graphs
# Generate data in DOT format to visualize decision trees
dot_data = export_graphviz(clf, out_file=None,
                           feature_names=FEATURE_NAME,
                           class_names=['[0]', '[1]',
                                        '[2]', '[3]', '[4]', '[5]'],
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)

# you can save the graph as a PDF file or display it on the screen.
graph.render('decision_tree_visualization',  format='png', view=False)

# cal_shap(model,X of dataset, Plot type what you want, unique actions)
cal_shap(clf,X,SHAP_PLOT_TYPE,actions)
#model: Model of distilated policy
#X_test: Test dataset
#SHAP_PLOT_TYPE: Decision_PLOT_TYPE
#actions: actions of test_dataset


