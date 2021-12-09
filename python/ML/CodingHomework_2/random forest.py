# 数据处理包
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 画图
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# 建模
from sklearn.preprocessing import scale, LabelEncoder  # 用于数据预处理模块的缩放器、标签编码
from sklearn.model_selection import train_test_split  # 数据集分类器 用于划分训练集和测试集
from sklearn.metrics import classification_report, accuracy_score  # 评估预测结果
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.ensemble import GradientBoostingClassifier  # XGB分类
# 设置输出全部结果 而非只有最后一个
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# 设置正常显示负号和中文
'''plt.rcParams['font.family'] = 'SimHei'  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号'''
#读取数据
data = pd.read_csv("D:/machine_learning/task2/Employee.csv")
print(data.shape) # 查看数据集结构
print(data.head()) # 预览数据
print(data.isnull().sum()) # 查看每一列的缺失值数量
data.info() # 查看特征类型
col_all = list(data.columns) # 全部特征
col_cate = ['Education','City','Gender','EverBenched'] # 分类特征
col_num = ['JoiningYear','PaymentTier','Age','ExperienceInCurrentDomain'] # 连续（数值）特征

data = pd.get_dummies(data=data,columns=["Education","City","Gender","EverBenched"],prefix_sep="+",drop_first=False)
print(data)

X = data.iloc[:,[0,1,2,3,5,6,7,8,9,10,11,12,13,14]].to_numpy()
print(X)
y = data['LeaveOrNot'].to_numpy()
X_scale = scale(X) # z使之服从标准正态分布
print (X_scale) # 查看缩放后的数据
print(X_scale.dtype)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4)
print(len(X_scale))

def get_subsample(dataSet,ratio):
    subdataSet=[]
    lenSubdata=round(len(dataSet)*ratio)
    while len(subdataSet) < lenSubdata:
        index=randrange(len(dataSet)-1)
        subdataSet.append(dataSet[index])
    print(subdataSet)
    return subdataSet

#选取任意的n个特征，在这n个特征中，选取分割时的最优特征
def get_best_spilt(dataSet,n_features):
    features=[]
    class_values=list(set(row[-1] for row in dataSet))
    b_index,b_value,b_loss,b_left,b_right=999,999,999,None,None
    while len(features) < n_features:
        index=randrange(len(dataSet[0])-1)
        if index not in features:
            features.append(index)
    #print 'features:',features
    for index in features:
        for row in dataSet:
            left,right=data_spilt(dataSet,index,row[index])
            loss=spilt_loss(left,right,class_values)
            if loss < b_loss:
                b_index,b_value,b_loss,b_left,b_right=index,row[index],loss,left,right
    #print b_loss
    #print type(b_index)
    return {'index':b_index,'value':b_value,'left':b_left,'right':b_right}
def build_tree(dataSet,n_features,max_depth,min_size):
    root=get_best_spilt(dataSet,n_features)
    sub_spilt(root,n_features,max_depth,min_size,1)
    return root
#创建随机森林
def random_forest(train,test,ratio,n_feature,max_depth,min_size,n_trees):
    trees=[]
    for i in range(n_trees):
        subTrain=get_subsample(train,ratio)
        tree=build_tree(subTrain,n_features,max_depth,min_size)
        #print 'tree %d: '%i,tree
        trees.append(tree)
    #predict_values = [predict(trees,row) for row in test]
    predict_values = [bagging_predict(trees, row) for row in test]
    return predict_values
#预测测试集结果
def predict(tree,row):
    predictions=[]
    if row[tree['index']] < tree['value']:
        if isinstance(tree['left'],dict):
            return predict(tree['left'],row)
        else:
            return tree['left']
    else:
        if isinstance(tree['right'],dict):
            return predict(tree['right'],row)
        else:
            return tree['right']




