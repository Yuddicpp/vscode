import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# for dirname, _, filenames in os.walk('Dataset_traffic-driving-style-road-surface-condition'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

df_opel_corsa_01 = pd.read_csv('Dataset_traffic-driving-style-road-surface-condition/opel_corsa_01.csv',sep='.',delimiter=';')

df_opel_corsa_02 = pd.read_csv('Dataset_traffic-driving-style-road-surface-condition/opel_corsa_02.csv',sep='.',delimiter=';')

df_peugeot_207_01 = pd.read_csv('Dataset_traffic-driving-style-road-surface-condition/peugeot_207_01.csv',sep='.',delimiter=';')

df_peugeot_207_02 = pd.read_csv('Dataset_traffic-driving-style-road-surface-condition/peugeot_207_02.csv',sep='.',delimiter=';')


# 使用skilearn
def Road_surface_data_process(data_raw):
    roadSurface_to_state = {
        'EvenPaceStyle' : 0,
        'AggressiveStyle.' : 1
    }
    y = data_raw['drivingStyle'].map(roadSurface_to_state).fillna(0).to_numpy()
    x = data_raw.iloc[:,:14].fillna(1).to_numpy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if type(x[i,j]) == str:
                x[i,j] = float(x[i,j].replace(',','.'))
    return x,y


X,Y = Road_surface_data_process(pd.concat([df_opel_corsa_01, df_opel_corsa_02, df_peugeot_207_01, df_peugeot_207_02], axis=0))
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
clf = GaussianNB()
clf.fit(x_train, y_train)
print(accuracy_score(clf.predict(x_test),y_test))