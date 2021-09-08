# This code is submitted by Ayanabha Ghosh (ayan-cs)

import pandas as pd
import numpy as np
df = pd.read_csv('diabetes.csv')

Y = np.array(df['Outcome'])
X = np.array([list(df.loc[i][:-1]) for i in range(len(df))])

for i in range(len(Y)):
  if Y[i] == 0:
    Y[i]=-1

size = len(Y)
random_indice = np.random.permutation(size)
num_train = int(size*0.7)
num_test = int(size*0.3)

X_train = X[random_indice[:num_train]]
y_train = Y[random_indice[:num_train]]
X_test = X[random_indice[-num_test:]]
y_test = Y[random_indice[-num_test:]]

import random
w = np.array([random.random() for _ in range(len(X_train[0]))])
print("Initial weight : ",w)
for i in range(num_train):
  inner_prod = np.inner(w, X_train[i])
  if y_train[i]*inner_prod <= 0 :
    w = np.add(w, np.dot(y_train[i], X_train[i]))
print("Updated weight : ",w)

y_pred=list()
for i in range(num_test):
  if np.inner(w, X_test[i]) > 0 :
    y_pred.append(1)
  else :
    y_pred.append(-1)
cclf=0
for i in range(num_test):
  if y_pred[i]==y_test[i]:
    cclf+=1
print("Accuracy = "+str(cclf/num_test))
