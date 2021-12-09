import numpy as np
from numpy.core.fromnumeric import size
from numpy.lib.function_base import place
from scipy.special import softmax
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class RNN():
    def __init__(self) -> None:
        self.U = np.array([[2.0,-1.0],[1.0,1.0]])
        self.V = np.array([0.5,1.0])
        self.W = np.array([[0.8,-0.1],[-0.12,0.8]])
        self.b = np.array([[0.2],[-0.1]])
        self.c = 0.25
        self.LR = 0.01
    
    def forward(self,x):
        T = x.shape[1]
        h = np.zeros((2,4))
        o = np.zeros(3)
        for t in range(T):
            h[:,t+1] = np.tanh(self.b + self.W.dot(h[:,t].reshape(2,1))+self.U.dot(x[:,t].reshape(2,1))).transpose()
            o[t] = self.c + self.V.dot(h[:,t+1].reshape(2,1))
        
        return h,o


    def BPTT(self,x,y):
        h,o = self.forward(x)
        
        dLdo = o-y
        dLdh = np.zeros((2,3))
        T = x.shape[1]
        for t in np.arange(T-1,-1,-1): # 2-0
            if t==2:
                dLdh[:,t] = self.V.transpose()*dLdo[t]
            else:
                dLdh[:,t] = self.W.transpose().dot(np.diag(1-(h[:,t+2])**2)).dot(dLdh[:,t+1].transpose()) + self.V.transpose()*dLdo[t]
        

        dLdc = 0
        dLdV = np.zeros(self.V.shape)
        dLdb = np.zeros(self.b.shape)
        dLdW = np.zeros(self.W.shape)
        dLdU = np.zeros(self.U.shape)
        for t in range(T):
            dLdc += dLdo[t]
            dLdV += dLdo[t]*h[:,t+1]
            dLdb += np.diag(1-(h[:,t+1])**2).dot(dLdh[:,t].transpose()).reshape(2,1)
            dLdW += np.diag(1-(h[:,t+1])**2).dot(dLdh[:,t].transpose()).reshape(2,1).dot(h[:,t].reshape(1,2))
            dLdU += np.diag(1-(h[:,t+1])**2).dot(dLdh[:,t].transpose()).reshape(2,1).dot(x[:,t].reshape(1,2))

        self.c -= self.LR * dLdc
        self.V -= self.LR * dLdV
        self.b -= self.LR * dLdb
        self.W -= self.LR * dLdW
        self.U -= self.LR * dLdU

        return mean_squared_error(o,y)
    
    def show_parameters(self):
        print(self.c,self.V,self.b,self.W,self.U)

    





if __name__ == "__main__":

    x = np.array([[1,-1,1],[2,0,-1]])
    y = np.array([-1,1,2])
    model = RNN()
    loss = []
    for i in range(1000):
        loss.append(model.BPTT(x,y))
    
    plt.plot(range(1000),loss)
    plt.show()
    model.show_parameters()
    
