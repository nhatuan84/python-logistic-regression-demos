import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

init = False
file = open('logistic_x.txt', 'rb')
for row in file:
    r = row.decode('utf8').strip().split(' ')
    if(init == False):
        x_train = np.array([[1], [np.float(r[0])], [np.float(r[len(r)-1])]])
        init = True
    else:
        x_train = np.append(x_train, [[1], [np.float(r[0])], [np.float(r[len(r)-1])]], axis=1);
init = False
file = open('logistic_y.txt', 'rb')
for row in file:
    if(init == False):
        y_train = np.array([[np.float(row.strip())]])
        init = True
    else:
        y_train = np.append(y_train, [[np.float(row.strip())]], axis=1);

m = y_train.shape[1]
theta = np.zeros((x_train.shape[0], 1))


def sigmoid(z):
	return 1/(1+np.exp(-z))

pos = np.flatnonzero(y_train == 1)
neg = np.flatnonzero(y_train == 0)

plt.plot(x_train[1, pos], x_train[2, pos], 'ro')
plt.plot(x_train[1, neg], x_train[2, neg], 'bo')  

yT = y_train.T
xT = x_train.T
#iterator 500 steps
for x in range(0, 10):
    h = sigmoid(theta.T.dot(x_train))
    error = h - y_train
    tmp = (-1)*y_train*np.log(h) - (1-y_train)*np.log((1-h))
    J = np.sum(tmp)/m;
    #calculate H
    H = (h*(1-h)*(x_train)).dot(x_train.T)/m
    #calculate dJ
    dJ = np.sum(error*x_train, axis=1)/m
    #gradient = H-1.dJ
    grad = inv(H).dot(dJ)
    #update theta
    theta = theta - (np.array([grad])).T
    print(J)
    
print(theta)

#plot boundary to separate data points

#u = np.linspace(np.ndarray.min(x_train[1:]), np.ndarray.max(x_train[1:]), num=2)
#v = np.linspace(np.ndarray.min(x_train[1:]), np.ndarray.max(x_train[1:]), num=2)
#z = np.zeros((len(u), len(v)))
#for i in range(0, len(u)):
#    for j in range(0, len(v)):
#        z[i, j] = theta[0][0] + theta[1][0]*u[i] + theta[2][0]*v[j]
#print(z)
#plt.contour(u, v, z)

plot_x = [np.ndarray.min(x_train[1:]), np.ndarray.max(x_train[1:])]
plot_y = np.subtract(np.multiply(-(theta[2][0]/theta[1][0]), plot_x), theta[0][0]/theta[1][0])
plt.plot(plot_x, plot_y, 'b-')

plt.show()
