# This code is submitted by Ayanabha (ayan-cs)

import pandas as pd
import numpy as np
df = pd.read_csv('/content/diabetes.csv')

Y = np.array(df['Outcome'])
X = np.array([list(df.loc[i][:-1]) for i in range(len(df))])

for i in range(len(Y)):
  if Y[i] == 0:
    Y[i]=-1

size = len(Y)
random_indice = np.random.permutation(size)
num_train = int(size*0.7)
num_test = int(size*0.3)
num_feature = len(X[0])

X_train = X[random_indice[:num_train]]
y_train = Y[random_indice[:num_train]]
X_test = X[random_indice[-num_test:]]
y_test = Y[random_indice[-num_test:]]

from cvxopt import matrix, solvers
C = 10
m,n = X_train.shape
y_train = y_train.reshape(-1, 1) * 1.
X_dash = y_train * X_train
H = np.dot(X_dash , X_dash.T) * 1.

P = matrix(H)
q = matrix(-np.ones((m, 1)))
G = matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
A = matrix(y_train.reshape(1, -1))
b = matrix(np.zeros(1))

sol = solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])
w = ((y_train * alphas).T @ X_train).reshape(-1,1)
print("Optimal weight : ",w)

cclf = 0
y_pred = np.inner(w.T, X_test)
for i in range(len(X_test)):
  if y_pred[0][i]*y_test[i]>1:
    cclf+=1
print("Accuracy : ",cclf/len(X_test)*100)
