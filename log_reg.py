import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

init = False
file = open('demo1x.dat', 'rb')
for row in file:
    r = row.strip().split(' ')
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
while True:
    J = 0
    #scan through training set
    for i in range(0, m):
        #for each training set
        x = x + 1;
        #calculate h_predicted
        h = sigmoid(theta*(xT[i].T));print(theta*(xT[i].T))
        error = h.T - yT[i]
        #accumulate error to J
        tmp = (-1)*y_train[i]*np.log(h) - (1-y_train[i])*np.log((1-h))
        J = J + tmp;
        #update theta for a training set
        theta = theta - 0.0001*(error*x_train[i]);
    J=J/m
    #plot J
    #update_line(g, x, J)
    print(J)
    if(abs(J)<0.0001):
        break
print(theta)
plt.show()


