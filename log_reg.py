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


def sigmoid(z):
    return 1/(1+np.exp(-z))

x = 0
xT = x_train.T
yT = y_train.T
preJ = 0
while True:
    J = 0
    for i in range(0, m):
        x = x + 1;
        h = sigmoid(theta.T.dot(x_train[:,i].T))
        error = h.T - yT[i]
        tmp = (-1)*yT[i]*np.log(h) - (1-yT[i])*np.log((1-h))
        J = J + tmp
        nX = np.array([x_train[:,i]]).T
        theta = theta - 0.001*(error*nX)
    J=J/m
    print(J)
    if(preJ == 0):
        preJ = J
    if(preJ < J):
        break
    else:
        preJ = J
print(theta)
#plt.show()

