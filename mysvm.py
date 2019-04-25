import numpy as np
import cvxopt as cvxopt
from cvxopt import solvers

class SVM:
    def __init__(self, l_multipliers, targets, k, data, sup_vectors, w_0):
        self.l_multipliers = l_multipliers
        self.targets = targets
        self.k = k
        self.data = data
        self.sup_vectors = sup_vectors
        self.w_0 = w_0

def train(data, targets, k, C=None):
    # Compute kernel matrix
    K = np.array([[k(x_1, x_2) for x_2 in data] for x_1 in data])
    # Compute P
    P = targets * targets.transpose() * K
    q = -np.ones((len(data), 1))
    G = -np.eye(len(data))
    h = np.zeros((len(data), 1))
    A = targets.reshape(1, len(data))
    A = A.astype(float)

    # We'll turn this into a vector in the following line when we pass
    # all these to qp
    b = 0.0

    # Solve the QP problem
    sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), cvxopt.matrix(
    G), cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))

    # Retreive the lagrange multipliers
    lagrange_multipliers = np.array(sol['x'])

    # Identify (the indices of) the support vectors
    threshold = 1e-5
    support_vectors = np.where(lagrange_multipliers > threshold)[0]

    # Compute the intercept
    w_0 = sum([targets[j] - sum([lagrange_multipliers[i] * targets[i] * k(data[i], data[j])
        for i in support_vectors])
           for j in support_vectors]) / len(support_vectors)

    complete_svm = SVM(lagrange_multipliers, targets, k, data, support_vectors, w_0)
    return complete_svm


# Binary classification
def classify(svm, inputs):
    classifications = [classifyOne(svm, x) for x in inputs]
    return classifications

# helper function for classify
def classifyOne(svm, input):
    return np.sign(sum([svm.l_multipliers[i] * svm.targets[i] * svm.k(svm.data[i], input) for i in svm.sup_vectors]) + svm.w_0)

