import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

init = False
file = open('demo1x.dat', 'rb')
for row in file:
    r = row.decode('utf8').strip().split(' ')
    if(init == False):
        x_train = np.array([[1], [np.float(r[0])], [np.float(r[len(r)-1])]])
        init = True
    else:
        x_train = np.append(x_train, [[1], [np.float(r[0])], [np.float(r[len(r)-1])]], axis=1);
init = False
file = open('demo1y.dat', 'rb')
for row in file:
    if(init == False):
        y_train = np.array([[np.float(row.strip())]])
        init = True
    else:
        y_train = np.append(y_train, [[np.float(row.strip())]], axis=1);

m = y_train.shape[1]
theta = np.array(np.zeros((x_train.shape[0], 1)))


def sigmoid(theta, x):
    return 1/(1+np.exp(theta.T.dot(x)))
    
yT = y_train.T
xT = x_train.T
#iterator 500 steps
for x in range(0, 2):
    h = sigmoid(theta, x_train)
    error = h.T - yT;
    tmp = (-1)*y_train*np.log(h) - (1-y_train)*np.log((1-h))
    J = tmp.dot(tmp.T)/m
    H = h.dot(1-h).dot(x).dot(x_train.T)/m
    theta = theta - inv(H)*x_train.dot(error)/m;
    print(J)
    
print(theta)
