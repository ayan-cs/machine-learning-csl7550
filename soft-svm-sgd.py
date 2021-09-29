# This code is submitted by Ayanabha (ayan-cs)

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
num_feature = len(X[0])

X_train = X[random_indice[:num_train]]
y_train = Y[random_indice[:num_train]]
X_test = X[random_indice[-num_test:]]
y_test = Y[random_indice[-num_test:]]

import numpy as np
import random
w = np.zeros(num_feature, dtype=np.float)

random_indices = np.random.permutation(train_size)
X_train = X[random_indices[:]]
y_train = y[random_indices[:]]

gamma = 0.0001
epoch = 100000
summ = [0. for _ in range(num_feature)]
for i in range(1, epoch+1):
    w = np.multiply(1/(gamma*i), summ)
    rand_instance = np.random.randint(train_size)
    h = y_train[rand_instance] * np.inner(w, X[rand_instance])
    if h < 1:
        summ = summ + y_train[rand_instance]*X_train[rand_instance]

print("Updated weight : ",w)

y_pred = np.inner(w, X_test)
cclf = 0
for i in range(len(y_pred)):
    if y_pred[i]*y_test[i] > 0:
        cclf+=1
print("Accuracy : ",cclf/len(y_pred)*100)
