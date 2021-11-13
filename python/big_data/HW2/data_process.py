from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
from time import *
from scipy.sparse import data
from sklearn.metrics.pairwise import cosine_similarity



df_train = pd.read_csv('Project2-data/netflix_train.txt',sep = ' ',names=['users','movie_title','rank','data'])
df_users = pd.read_csv('Project2-data/users.txt',names = ['users'])

data_train = pd.DataFrame(0,index = np.array(df_users['users']),columns=np.array(range(1,10001)))


for index, row in df_train.iterrows():
    print(index,row['rank'])
    data_train.loc[row['users'],row['movie_title']] = row['rank']

print(data_train)
data_train.to_csv("data_train.csv")





