import os,re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PATH = '10.21数据'



def loc_show(loc,data):
    print(loc)
    x = data.iloc[:,0].to_numpy()
    y = data.iloc[:,1].to_numpy()
    plt.xlim(xmax=5,xmin=-5)
    plt.ylim(ymax=5,ymin=-5)
    plt.scatter(x,y)
    plt.scatter(loc[0],loc[1],c='red')
    plt.scatter(0,0,c='black')
    plt.pause(1)
    plt.close()
    


plt.ion()
for dirname, _, filenames in os.walk(PATH):
    for filename in filenames:
        if(re.search(r'xlsx',filename)):
            print(filename)
            loc = re.findall(r'-?\d+\.?\d*',filename)
            loc = [ float(x) for x in loc ]
            data = pd.read_excel(os.path.join(dirname, filename))
            plt.figure(filename)
            loc_show(loc,data)

plt.ioff()
plt.show()