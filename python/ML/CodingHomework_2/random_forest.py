import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing



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
label = data_raw.iloc[:,8].to_numpy()

data['Education'] = data['Education'].map(Education_to_state)
data['City'] = data['City'].map(City_to_state)
data['Gender'] = data['Gender'].map(Gender_to_state)
data['EverBenched']= data['EverBenched'].map(EverBenched_to_state)
data['JoiningYear'] = data['JoiningYear']
data['Age'] = data['Age']

# print(data)
data = data.to_numpy()

train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=0.1,shuffle=True)

model = RandomForestClassifier(n_estimators=20,bootstrap = True,max_features = 'sqrt')
model.fit(train_data, train_label)

rf_predictions = model.predict(test_data)

roc_value = roc_auc_score(test_label, rf_predictions)
print(roc_value)

