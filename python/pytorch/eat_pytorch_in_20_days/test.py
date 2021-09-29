import os, sys

from numpy.core.numeric import indices
os.chdir(sys.path[0])

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import torch 
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset


# Series
# data = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
# print(data)

# Dataframe
# d= {
#     "one": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
#     "two": pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"]),
# }
# data = pd.DataFrame(d)
# print(data.columns)
# df2 = pd.DataFrame(
#     {
#         "A": 1.0,
#         "B": pd.Timestamp("20130102"),
#         "C": pd.Series(1, index=list(range(4)), dtype="float32"),
#         "D": np.array([3] * 4, dtype="int32"),
#         "E": pd.Categorical(["test", "train", "test", "train"]),
#         "F": "foo",
#     }
# )
# print(df2.iloc[:,0:1])

dftrain_raw = pd.read_csv('./data/titanic/train.csv')
dftest_raw = pd.read_csv('./data/titanic/test.csv')

print("test")
print(dftest_raw.columns)
def preprocessing(dfdata):

    dfresult= pd.DataFrame()

    #Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns ]
    dfresult = pd.concat([dfresult,dfPclass],axis = 1)

    #Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult,dfSex],axis = 1)

    #Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    #SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    #Carbin
    dfresult['Cabin_null'] =  pd.isna(dfdata['Cabin']).astype('int32')

    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)

    return(dfresult)

# x_train = preprocessing(dftrain_raw).values
# y_train = dftrain_raw[['Survived']].values

# x_test = preprocessing(dftest_raw).values
# y_test = dftest_raw[['Survived']].values

# print("x_train.shape =", x_train.shape )
# print("x_test.shape =", x_test.shape )

# print("y_train.shape =", y_train.shape )
# print("y_test.shape =", y_test.shape )

# print(x_train)

