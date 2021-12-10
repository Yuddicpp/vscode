import pandas as pd
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import tree
import random
import math
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier



def age_to(x):
    if(x<23):
        return 0
    elif(x<25):
        return 1
    elif(x<26):
        return 2
    elif(x<27):
        return 3
    elif(x<28):
        return 4
    elif(x<31):
        return 5
    elif(x<36):
        return 6
    # elif(x<35):
    #     return 7
    # elif(x<37):
    #     return 8
    # elif(x<39):
    #     return 9
    else:
        return 7

def data_process():
    """
    process the data into train/test data
    """

    Education_to_state = {
        'Bachelors': 0,
        'Masters': 1,
        'PHD': 2
    }
    City_to_state = {
        'Bangalore': 0,
        'Pune': 1,
        'New Delhi': 2,

    }
    Gender_to_state = {
        'Male': 0,
        'Female': 1
    }
    EverBenched_to_state = {
        'No': 0,
        'Yes': 1
    }


    data_raw = pd.read_csv("Employee.csv")

    data = data_raw.iloc[:,0:8]
    label = data_raw.iloc[:,8]

    data['Education'] = data['Education'].map(Education_to_state)
    data['City'] = data['City'].map(City_to_state)
    data['Gender'] = data['Gender'].map(Gender_to_state)
    data['EverBenched']= data['EverBenched'].map(EverBenched_to_state)
    data['JoiningYear'] = data['JoiningYear'] - data['JoiningYear'].min()
    # data['Age'] = data['Age'] - data['Age'].min()
    data['Age'] = data['Age'].map(age_to)
    # print(data['Age'])

    # print(data)
    data = data.to_numpy()
    label = label.to_numpy()
    train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=0.1,shuffle=True)
    return train_data,test_data,train_label,test_label 
    # return data,label

def bootstrap_sampling(train_data,train_label):
    l = train_data.shape[0]
    train_data_result = np.zeros(train_data.shape)
    train_label_result = np.zeros(train_label.shape)

    for i in range(l):
        k = math.floor(random.random()*l)
        # print(k,l)
        train_data_result[i] = train_data[k]
        train_label_result[i] = train_label[k]
    return train_data_result,train_label_result


train_data,test_data,train_label,test_label  = data_process()

estimator = 20
result = []
for i in range(estimator):
    model = tree.DecisionTreeClassifier(criterion="gini",splitter="random")
    model.fit(bootstrap_sampling(train_data,train_label)[0],bootstrap_sampling(train_data,train_label)[1])
    result.append(model.predict(test_data))


result = np.array(result)
label = np.zeros(result.shape[1])
for i in range(len(label)):
    l = result[:,i].astype(np.int64)
    label[i] = np.argmax(np.bincount(l))
    # print(label[i])

Accuracy = metrics.accuracy_score(label,test_label)
Precision = metrics.precision_score(label,test_label)
Recall = metrics.recall_score(label,test_label)

print(Accuracy,Precision,Recall)






# model = RandomForestClassifier(n_estimators=20,bootstrap = True,max_features = 'sqrt')
# model.fit(train_data, train_label)

# rf_predictions = model.predict(test_data)

# roc_value = metrics.roc_auc_score(test_label, rf_predictions)
# print(roc_value)
