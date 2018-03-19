import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

#we read data from files line by line
init = False
file = open('ex5Logx.dat', 'rb')
for row in file:
    r = row.decode('utf8').strip().split(',')
    if(init == False):
        x1 = np.array([np.float(r[0])])
        x2 = np.array([np.float(r[1])])
        init = True
    else:
        x1 = np.append(x1, [np.float(r[0])])
        x2 = np.append(x2, [np.float(r[1])])

init = False
file = open('ex5Logy.dat', 'rb')
for row in file:
    if(init == False):
        y_train = np.array([[np.float(row.strip())]])
        init = True
    else:
        y_train = np.append(y_train, [[np.float(row.strip())]], axis=1);

def sigmoid(z):
	return 1/(1+np.exp(-z))


def create_feature(x1, x2):
    x_train = np.array([np.ones(len(x1))])
    for total in range(1, 7):
        for p in range(total, -1, -1):
            x_train = np.append(x_train, [np.power(x1, p)*np.power(x2, total-p)], axis=0)
    return x_train
            
x_train = create_feature(x1, x2)
#number of training examples
m = y_train.shape[1]
#init theta
theta = np.array(np.zeros((x_train.shape[0], 1)))

pos = np.flatnonzero(y_train == 1)
neg = np.flatnonzero(y_train == 0)

plt.plot(x_train[1, pos], x_train[2, pos], 'ro')
plt.plot(x_train[1, neg], x_train[2, neg], 'bo') 

#modify lamda to see results
lamda = 1

one = np.identity(x_train.shape[0])
one[0,0] = 0

for x in range(0, 15):
    h = sigmoid(theta.T.dot(x_train))
    error = h - y_train
    tmp = (-1)*y_train*np.log(h) - (1-y_train)*np.log((1-h))
    J = np.sum(tmp)/m + lamda*theta[1::,0].T.dot(theta[1::,0])/2*m
    #calculate H with regularization from 1
    H = (h*(1-h)*(x_train)).dot(x_train.T)/m + lamda*one/m
    #calculate dJ with regularization from 1
    dJ = np.sum(error*x_train, axis=1)/m
    dJ[1:] = dJ[1:] - lamda*theta.T[0,1::]/m
    #gradient = H-1.dJ
    grad = inv(H).dot(dJ)
    #update theta
    theta = theta - (np.array([grad])).T
    print(J)

print(theta)

#plot boundary 
u = np.linspace(-1, 1.5, 200)
v = np.linspace(-1, 1.5, 200)
#create input grid
uu, vv = np.meshgrid(u, v, sparse=False)
#output contour
plotz = np.zeros(uu.shape)
#evaluate z
for i in range(0, len(uu)):	
	z = theta.T.dot(create_feature(uu[i], vv[i]))
	for j in range(0,z.shape[1]):
		plotz[i][j] = z[0][j]
#plot contour
plt.contourf(uu, vv, plotz, linestyles='dashed')
plt.show()
