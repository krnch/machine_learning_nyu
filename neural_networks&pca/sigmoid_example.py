import numpy as np
import math
def sigmoid(x):

  	return 1 / (1 + np.exp(-x))



x = np.array([12,3,4])

print (x)
print (sigmoid(x))

print np.eye(8,8)


#https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.eig.html
from numpy import linalg as LA

w, v = LA.eig(np.array([[0, 6], [6, 3]]))
print w
print v

import numpy as np
a = np.array([[8,5,3],[2, 8, 10],[6,0,1],[8, 2, 6]]).T
print a
b = np.cov(a)
print(b)
w1, v1 = LA.eig(b)
print w1
print v1