# Code submitted by AyanG (ayan-cs)

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data[:100]
y = iris.target[:100]

for i in range(len(y)):
  if y[i] == 0:
    y[i] = -1

size = len(X)
num_feature = len(X[0])
train_size = int(size*0.7)

X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

import numpy as np
from cvxopt import matrix, solvers
P = matrix(np.identity(num_feature, dtype=np.float))
q = matrix(np.zeros((num_feature,), dtype=np.float))
G = matrix(np.zeros((train_size, num_feature), dtype=np.float))
h = - matrix(np.ones((train_size,), dtype=np.float))

for i in range(train_size) :
  G[i,:] = - X_train[i,:]*y_train[i]

sol = solvers.qp(P, q, G, h)
print(sol['x'])

w = np.zeros((num_feature,), dtype=np.float)
for i in range(num_feature):
  w[i]=sol["x"][i]
print("Optimal weights : ",w)

cclf = 0
y_pred = np.inner(w, X_test)
for i in range(len(X_test)):
  if y_pred[i]*y_test[i]>1:
    cclf+=1
print("Accuracy : ",cclf/len(X_test))
