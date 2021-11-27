import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score



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
data['JoiningYear'] = data['JoiningYear'] - data['JoiningYear'].min()
data['Age'] = data['Age'] - data['JoiningYear'].min()

data = data.to_numpy()

train_data = data[:4000,:]
train_label = label[:4000]

test_data = data[4000:,:]
test_label = label[4000:]

model = RandomForestClassifier(n_estimators=10,bootstrap = True,max_features = 'sqrt')
model.fit(train_data, train_label)

rf_predictions = model.predict(test_data)

roc_value = roc_auc_score(test_label, rf_predictions)
print(roc_value)

