import os
from IPython.core.display import display, display_png
from sklearn import tree
import graphviz
import numpy as np
import pydotplus
from IPython.display import Image
from sklearn.tree import DecisionTreeRegressor

x= np.array([0,0.1,0.2,0.3,0.4,0.45,0.5,0.6,0.7,0.8,0.9,0.95,1])
x = x.reshape(-1,1)
y = np.array([4,2.4,1.5,1,1.2,1.5,1.8,2.6,3,4,4.5,5,6])
# 建立模型
clf_tree = DecisionTreeRegressor(min_impurity_decrease=0.005)
# 训练决策树模型
clf_tree.fit(x,y)

print(clf_tree.predict([[0.76]]))
dot_data = tree.export_graphviz(clf_tree, out_file=None,  
                         filled=True, rounded=True,  
                         special_characters=True)
 
graph = pydotplus.graph_from_dot_data(dot_data)  
img = Image(graph.create_png())
graph.write_png("out.png")
