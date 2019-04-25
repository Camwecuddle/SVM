import numpy as np
import cvxopt as cvxopt
from cvxopt import solvers

# The code that accompanies Marsland, "Machine Learning: An Algorithmic
# Perspective" was useful in preparing this.

# Linear kernel


def linear(x1, x2):
    return 1 + np.dot(x1, x2)


# Six points on the x axis
X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
              [-1.0, 0.0], [-2.0, 0.0], [-3.0, 0.0]])

# Points on the positive x axis are in one class, negative in the other
Y = np.array([[1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0]])

# Make the kernel matrix
K = np.array([[linear(x_1, x_2) for x_2 in X] for x_1 in X])
print("K = " + str(K))


# This is what I think the computation of P  should be (actual matrix
# multiplication between Y*Y.transpose() and K), but it doesn't work.
#P = np.matmul(Y * Y.transpose(), K)
# print(P)

# The following apparently does component-wise (entrywise) matrix multiplication
# (also called Hadamard product), not what is usually called
# matrix multiplication.
P = Y * Y.transpose() * K
print("P")
print(P)

# Make the other  vectors and matrices
q = -np.ones((6, 1))
print("q")
print(q)

G = -np.eye(6)
print("G")
print(G)

h = np.zeros((6, 1))
print("h")
print(h)

A = Y.reshape(1, 6)
print("A")
print(A)

# We'll turn this into a vector in the following line when we pass
# all these to qp
b = 0.0

# Solve the QP problem
sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), cvxopt.matrix(
    G), cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))

# Retreive the lagrange multipliers
lagrange_multipliers = np.array(sol['x'])
print("lagrange_multipliers")
print(lagrange_multipliers)

# Identify (the indices of) the support vectors
threshold = 1e-5
support_vectors = np.where(lagrange_multipliers > threshold)[0]
print("support_vectors")
print(support_vectors)

# Compute the intercept
w_0 = sum([Y[j][0] - sum([lagrange_multipliers[i] * Y[i][0] * linear(X[i], X[j])
                          for i in support_vectors])
           for j in support_vectors]) / len(support_vectors)
print("w_0")
print(w_0)

# Function to classify datapoints (note that this relies on
# global variables: lagrange_multipliers, Y, X, support_vectors, and w_0


def classify(x):
    return np.sign(sum([lagrange_multipliers[i] * Y[i][0] * linear(X[i], x)
                        for i in support_vectors]) + w_0)


# Some other points, as a test set
X_test = [[4.0, 5.0], [1.5, 0.0], [7.5, 10.0],
          [-5.0, 0.0], [-2.0, -4.0], [-2.0, 4.0]]

# Classify
for x_i in X:
    print(str(x_i) + ": " + str(classify(x_i)))

for x_i in X_test:
    print(str(x_i) + ": " + str(classify(x_i)))
