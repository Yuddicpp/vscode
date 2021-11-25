# import numpy as np
# from numpy.linalg import inv
# import torch 
# from torch import nn 
# from sklearn import svm
# import matplotlib.pyplot as plt



# # X = np.array([[0,0],[1,0],[0,1],[1,1],[1,2],[2,1],[2,2]])
# # Y = np.array([-1,-1,-1,1,1,1,1])

# # X = np.array([[0,0],[0,1],[1,0],[1,1]])
# # Y = np.array([-1,1,1,-1])

# X = np.array([[-1,1],[-1,-1],[1,-1],[1,1]])
# Y = np.array([-1,1,1,-1])

# clf = svm.SVC(kernel="linear",C=1000)
# clf.fit(X,Y)
# print(clf.n_support_)
# print(clf.get_params())

# plt.scatter(X[:, 0], X[:, 1], c=Y, s=30, cmap=plt.cm.Paired)

# # plot the decision function
# ax = plt.gca()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()

# # create grid to evaluate model
# xx = np.linspace(xlim[0], xlim[1], 30)
# yy = np.linspace(ylim[0], ylim[1], 30)
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = clf.decision_function(xy).reshape(XX.shape)

# # plot decision boundary and margins
# ax.contour(
#     XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
# )
# # plot support vectors
# ax.scatter(
#     clf.support_vectors_[:, 0],
#     clf.support_vectors_[:, 1],
#     s=100,
#     linewidth=1,
#     facecolors="none",
#     edgecolors="k",
# )
# plt.show()

# from math import log2

# N = 9
# N1 = 3
# N2 = 6
# N3 = 3
# P = 9
# N1P = 3/N1
# N2P = 0/N2
# N3P = 2/N3

# ans = -(N1/N)*(N1P*log2(N1P)+(1-N1P)*log2(1-N1P))-(N2/N)*(N2P*log2(N2P)+(1-N2P)*log2(1-N2P))-(N3/N)*(N3P*log2(N3P)+(1-N3P)*log2(1-N3P))
# ans = -(N1/N)*log2(N1/N)-(N2/N)*log2(N2/N)-(N3/N)*log2(N3/N)
# ans = -(N1/N)*log2(N1/N)-(N2/N)*log2(N2/N)
# ans = -(N1/N)*(N1P*log2(N1P)+(1-N1P)*log2(1-N1P))-(N3/N)*(N3P*log2(N3P)+(1-N3P)*log2(1-N3P))
# ans = (N1/N)*2*N1P*(1-N1P)+(N2/N)*2*N2P*(1-N2P)
# print(ans)

import numpy as np

x = np.array([0,0.1,0.2,0.3,0.4,0.45,0.5,0.6,0.7,0.8,0.9,0.95,1.0])
y = np.array([4,2.4,1.5,1.0,1.2,1.5,1.8,2.6,3.0,4.0,4.5,5.0,6.0])

# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([5.56,5.7,5.91,6.4,6.8,7.05,8.9,8.7,9,9.05])

a0 = 0.05
m = 0
for i in range(12):
    if(a0<x[i+1] and a0>x[i]):
        m = i+1
print(y[:m],y[m:])
print(np.mean(y[:m]),np.mean(y[m:]))
print(np.std(y[:m]),np.std(y[m:]))
