# This code is submitted by Ayanabha Ghosh (ayan-cs)

import pandas as pd
import numpy as np
df = pd.read_csv('./datasets/diabetes.csv')

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

# yixi matrix (not inner product)
res = np.array([[0 for a in range(len(X_train[0]))] for b in range(num_train)])
for i in range(num_train):
  for j in range(len(X_train[0])):
    res[i][j] = y_train[i] * X_train[i][j]

import pulp as p

Lp_prob = p.LpProblem('HSLP',p.LpMinimize)
Lp_prob+=1

w = np.array([p.LpVariable('w'+str(i)) for i in range(len(X_train[0]))])
w = np.transpose(w)
inner_prod = list(np.inner(w,res)) # <w, yixi>
for i in inner_prod:
  Lp_prob+=i>=1

status = Lp_prob.solve()
print(p.LpStatus[status])

for i in range(len(X_train[0])):
  print('w'+str(i)+' = '+str(p.value(w[i])))

w_val = [p.value(w[i]) for i in range(len(w))]
y_pred = list()
for i in range(num_test):
  if np.inner(w_val, X_test[i])>0:
    y_pred.append(1)
  else :
    y_pred.append(-1)

cclf=0
for i in range(num_test):
  if y_pred[i]==y_test[i]:
    cclf+=1
print("Accuracy = "+str(cclf/num_test))
