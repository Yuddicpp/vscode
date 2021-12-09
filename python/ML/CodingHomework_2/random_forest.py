import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
import numpy as np


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
    data['Age'] = data['Age'] - data['Age'].min()

    # print(data)
    # data = data.to_numpy()
    # train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=0.1,shuffle=True)
    # return train_data,test_data,train_label,test_label 
    return data,label



class TreeNode:

    def __init__(self, left, right, feature, X_data, Y_data):
        self.lef = left
        self.right = right
        self.feature = feature
        self.X_data = X_data
        self.Y_data = Y_data


class DecisionTreeCART:

    def __init__(self, X, Y):
        self.X_train = X
        self.Y_train = Y
        self.root_node = TreeNode(None,None,None,X,Y)
        self.classification(self.root_node,self.X_train,self.Y_train)

    def get_features(self, X_train_data):
        features = dict.fromkeys(X_train_data.columns)
        for i in range(len(X_train_data.columns)):
            feature = X_train_data.columns[i]
            features[feature] = list(X_train_data[feature].value_counts().keys())

        return features
    
    def classification(self,node,x,y):
        features = self.get_features(x)
        gini = self.compute_gini(features,x,y)
        max_gini = (max(max(gini.values())))
        for feature,feature_item in features:
            for i in range(len(feature_item)):
                if gini[feature][i] == max_gini:
                    split_feature = feature
                    split_feature_item = features[feature][i]
        node.feature = { split_feature : split_feature_item}
        data = x.join(y)
        left = data.drop(split_feature)
        right = data.loc[data[split_feature]!=split_feature_item]
        left_len = left.shape[1]
        right_len = right.shape[1]
        node_left = TreeNode(None,None,None,left[:,0:left_len-1],left[:,left_len-1])
        node_right = TreeNode(None,None,None,right[:,0:right_len-1],right[:,right_len-1])
        node.left = node_left
        node.right = node_right



    def compute_gini(self, features, x, y):
        user_nums = x.shape[0]
        gini = dict.fromkeys(x.columns)
        data = x.join(y)
        for feature in x.columns:
            gini_feature = []
            for item in features[feature]:
                user_feature = x.loc[x[feature] == item].shape[0]
                user_feature_y = data.loc[x[feature] == item].loc[data['LeaveOrNot'] == 1].shape[0]
                other_user_feature_y = data.loc[x[feature] != item].loc[data['LeaveOrNot'] == 1].shape[0]
                gini_feature.append((user_feature/user_nums)*2*(user_feature_y/user_feature)*(1-user_feature_y/user_feature)+(1-user_feature/user_nums)*2*(other_user_feature_y/(user_nums-user_feature))*(1-other_user_feature_y/(user_nums-user_feature)))
            
            gini[feature] = gini_feature
        return gini

        
        

data,label = data_process()

cart = DecisionTreeCART(data,label)
# print(cart.get_features(data))












# train_data,test_data,train_label,test_label  = data_process()

# model = RandomForestClassifier(n_estimators=20,bootstrap = True,max_features = 'sqrt')
# model.fit(train_data, train_label)

# rf_predictions = model.predict(test_data)

# roc_value = roc_auc_score(test_label, rf_predictions)
# print(roc_value)



