import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

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
        'SmoothCondition' : 0,
        'FullOfHolesCondition' : 1,
        'UnevenCondition' : 2
    }
    y = data_raw['roadSurface'].map(roadSurface_to_state).to_numpy()
    x = data_raw.iloc[:,:14].fillna(0).to_numpy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if type(x[i,j]) == str:
                x[i,j] = float(x[i,j].replace(',','.'))
    print(x,y)
    return x,y

X,Y = Road_surface_data_process(pd.concat([df_opel_corsa_01, df_opel_corsa_02, df_peugeot_207_01, df_peugeot_207_02], axis=0))
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit(x_train)

x_train_transformed = min_max_scaler.transform(x_train)
x_test_transformed = min_max_scaler.transform(x_test)

clf = LogisticRegression(random_state=0,max_iter=20000).fit(x_train_transformed, y_train)
print(clf.score(x_test_transformed,y_test))
