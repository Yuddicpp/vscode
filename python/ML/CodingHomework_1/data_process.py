import pandas as pd
import os
import numpy as np

for dirname, _, filenames in os.walk('Dataset_traffic-driving-style-road-surface-condition'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df_opel_corsa_01 = pd.read_csv('Dataset_traffic-driving-style-road-surface-condition/opel_corsa_01.csv',delimiter=';')

df_opel_corsa_02 = pd.read_csv('Dataset_traffic-driving-style-road-surface-condition/opel_corsa_02.csv',delimiter=';')

df_peugeot_207_01 = pd.read_csv('Dataset_traffic-driving-style-road-surface-condition/peugeot_207_01.csv',delimiter=';')

df_peugeot_207_02 = pd.read_csv('Dataset_traffic-driving-style-road-surface-condition/peugeot_207_02.csv',delimiter=';')

print(df_opel_corsa_01 )