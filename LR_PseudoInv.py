#This code is submitted by Ayanabha Ghosh (ayan-cs)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./datasets/insurance.csv')
print(df.head(5))
le = LabelEncoder()
le.fit(df['sex'])
df['sex'] = le.transform(df['sex'])
le.fit(df['smoker'])
df['smoker'] = le.transform(df['smoker'])
le.fit(df['region'])
df['region'] = le.transform(df['region'])
print(df.head(5))

X = np.array([df.loc[i][:-1] for i in range(len(df))])
Y = np.array(df['expenses'])

size = len(df)
random_indice = np.random.permutation(size)
num_train = int(size*0.7)
num_test = int(size*0.3)

X_train = X[random_indice[:num_train]]
y_train = Y[random_indice[:num_train]]
X_test = X[random_indice[-num_test:]]
y_test = Y[random_indice[-num_test:]]

if len(X_train)!=len(X_train[0]) or np.linalg.det(X_train)==0:
  w1 = np.matmul(np.linalg.pinv(X_train), np.transpose(y_train))
else :
  w1 = np.matmul(np.transpose(y_train), np.linalg.inv(X_train))

y_pred = np.inner(w1, X_test)
u = sum((y_pred - y_test)**2)
v = sum((y_test - np.mean(y_test))**2)
r2 = 1 - (u/v)
print("Score : "+str(r2))

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, y_train)
print("Score : "+str(clf.score(X_test, y_test)))
