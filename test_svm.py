import numpy as np
from sklearn.metrics.pairwise import rbf_kernel as sklrbf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mysvm

# Kernel function from TVD code for testing
def linear(x1, x2):
    return 1 + np.dot(x1, x2)

def rbf(x1, x2) :
    return sklrbf([x1], [x2])[0][0]

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33, random_state=42)

for i in range(len(y_train)):
    if(y_train[i] > 0):
        y_train[i] = 1.0
    else:
        y_train[i] = -1.0
    print(y_train[i])

for i in range(len(y_test)):
    if(y_test[i] > 0):
        y_test[i] = 1.0
    else:
        y_test[i] = -1.0
    print(y_test[i])



# Six points on the x axis
X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0],
              [-1.0, 0.0], [-2.0, 0.0], [-3.0, 0.0], [-4.0, 0.0]])

# Points on the positive x axis are in one class, negative in the other
Y = np.array([[1.0], [1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0], [-1.0]])

# Some other points, as a test set
# X_test = [[4.0, 5.0], [1.5, 0.0], [7.5, 10.0],
#           [-5.0, 0.0], [-2.0, -4.0], [-2.0, 4.0]]

# y_test = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]


def accuracy(trained_svm, X_test, y_test):
    score = 0
    classified = mysvm.classify(trained_svm, X_test)
    for i in range(len(classified)):
        print(str(y_test[i]) + ": classified as :" + str(classified[i][0]))

        if(classified[i] == y_test[i]):
            score += 1
    print("Score: " + str(score) + "/" + str(len(y_test)) + "  " + str((score/len(y_test)) * 100) + "%")

# # Classify
# for x_i in X:
#     print(str(x_i) + ": " + str(mysvm.classifyOne(trained_svm, x_i)))

# for x_i in X_test:
#     print(str(x_i) + ": " + str(mysvm.classifyOne(trained_svm, x_i)))


# Test on toy data set
# mysvm.classify(mysvm.train(X_train, y_train, linear), X_test)


# trained_svm = mysvm.train(X, Y, linear)
# accuracy(trained_svm, X_test, y_test)

trained_svm = mysvm.train(X_train, y_train, rbf)
accuracy(trained_svm, X_test, y_test)

