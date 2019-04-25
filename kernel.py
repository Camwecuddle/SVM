import numpy as np
from sklearn.metrics.pairwise import rbf_kernel as sklrbf

def linear(x1, x2) :
    return 1 + np.dot(x1, x2)

def make_poly_kernel(s) :
    return lambda x1, x2 : (1 + np.dot(x1, x2))**s

def rbf(x1, x2) :
    return sklrbf([x1], [x2])[0][0]

x1 = [1.0, 35.7, 47.3]
x2 = [9.7, 20.8, 37.7]
poly = make_poly_kernel(3)

print(linear(x1, x2))
print(poly(x1, x2))
print(rbf(x1, x2))
