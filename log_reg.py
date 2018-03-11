import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

x = []
y = []

file = open('demo1x.dat', 'rb')
for row in file:
	r = row.strip().split(' ')
	x.append([np.float(r[0]), np.float(r[len(r)-1])])

file = open('demo1y.dat', 'rb')
y = [np.float(row.strip()) for row in file]

