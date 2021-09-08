# This code is submitted by Ayanabha Ghosh (ayan-cs)

import pandas as pd
import numpy as np
df = pd.read_csv('diabetes.csv')

Y = np.array(df['Outcome'])
X = np.array([list(df.loc[i][:-1]) for i in range(len(df))])

import numpy as np
size = len(Y)
random_indice = np.random.permutation(size)
num_train = int(size*0.7)
num_test = int(size*0.3)

X_train = X[random_indice[:num_train]]
y_train = Y[random_indice[:num_train]]
X_test = X[random_indice[-num_test:]]
y_test = Y[random_indice[-num_test:]]

def sigmoid(z):
  return 1/(1+np.exp(-z))

import random
epoch = 1000000
alpha = 0.00015
w = [0 for _ in range(len(X_train[0]))]
b=0
print("Initial weight : ",w)
num_features = len(X_train[0])
for i in range(epoch):
  z = np.dot(X_train, np.transpose(w))+b
  h = sigmoid(z)
  cost = -(1/num_features)*(y_train*np.log(h) + (1-y_train)*np.log(1-h))
  d_cost = (1/num_features)*np.dot(np.transpose(X_train), (h-y_train))
  d_b = (1/num_features)*(h-y_train)
  w = w - alpha*np.transpose(d_cost)
  b = b - alpha*b
print("Updated weight : ",w)

z_pred = np.dot(X_test, np.transpose(w))+b
h_pred = sigmoid(z_pred)

cclf = 0
for i in range(num_test) :
  if y_test[i]==1 and h_pred[i]>0.5:
    cclf+=1
  if y_test[i]==0 and h_pred[i]<0.5:
    cclf+=1

print("Accuracy : "+str(cclf/num_test))
