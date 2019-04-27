import numpy as np
from sklearn.metrics.pairwise import rbf_kernel as sklrbf
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import mysvm

# Kernel function from TVD code for testing
def linear(x1, x2):
    return 1 + np.dot(x1, x2)

def rbf(x1, x2) :
    return sklrbf([x1], [x2])[0][0]

def make_poly_kernel(s) :
    return lambda x1, x2 : (1 + np.dot(x1, x2))**s

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33, random_state=42)

data2 = load_breast_cancer()
X_train2, X_test2, y_train2, y_test2 = train_test_split(data2.data, data2.target, test_size=0.33, random_state=42)

for i in range(len(y_train)):
    if(y_train[i] > 0):
        y_train[i] = 1.0
    else:
        y_train[i] = -1.0
    # print(y_train[i])

for i in range(len(y_test)):
    if(y_test[i] > 0):
        y_test[i] = 1.0
    else:
        y_test[i] = -1.0
    # print(y_test[i])

# breast cancer data set
for i in range(len(y_train2)):
    if(y_train2[i] == 0):
        y_train2[i] = -1.0
    else:
        y_train2[i] = 1.0
    # print(y_train[i])

for i in range(len(y_test)):
    if(y_test2[i] == 0):
        y_test2[i] = -1.0
    else:
        y_test2[i] = 1.0
    # print(y_test[i])


# Six points on the x axis
X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0],
              [-1.0, 0.0], [-2.0, 0.0], [-3.0, 0.0], [-4.0, 0.0]])

# Points on the positive x axis are in one class, negative in the other
Y = np.array([[1.0], [1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0], [-1.0]])

# Some other points, as a test set
X_test2 = [[4.0, 5.0], [1.5, 0.0], [7.5, 10.0],
           [-5.0, 0.0], [-2.0, -4.0], [-2.0, 4.0]]

y_test2 = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]


def accuracy(trained_svm, X_test, y_test):
    score = 0
    classified = mysvm.classify(trained_svm, X_test)
    for i in range(len(classified)):
        # print(str(y_test[i]) + ": classified as :" + str(classified[i][0]))

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

trained_iris_svm = mysvm.train(X_train, y_train, linear, 1)
print("Iris Data Set: Distinguishing target 0 from target 1 and 2.")
print("linear Kernel Function")
accuracy(trained_iris_svm, X_test, y_test)

trained_iris_svm = mysvm.train(X_train, y_train, rbf)
print("Iris Data Set: Distinguishing target 0 from target 1 and 2.")
print("rbf Kernel Function")
accuracy(trained_iris_svm, X_test, y_test)

trained_iris_svm = mysvm.train(X_train, y_train, make_poly_kernel(3), 1)
print("Iris Data Set: Distinguishing target 0 from target 1 and 2.")
print("make poly Kernel Function")
accuracy(trained_iris_svm, X_test, y_test)
print("Changing C doesn't affect these classifications")

# breast cancer data set
trained_breast_cancer_svm = mysvm.train(X_train2, y_train2, linear, 1)
print("Breast cancer Data Set: Distinguishing target 0 from target 1 and 2.")
print("linear Kernel Function")
accuracy(trained_breast_cancer_svm, X_test2, y_test2)

trained_breast_cancer_svm = mysvm.train(X_train2, y_train2, rbf)
print("Breast cancer Data Set: Distinguishing target 0 from target 1 and 2.")
print("rbf Kernel Function")
accuracy(trained_breast_cancer_svm, X_test2, y_test2)

trained_breast_cancer_svm = mysvm.train(X_train2, y_train2, make_poly_kernel(3), 1)
print("Breast cancer Data Set: Distinguishing target 0 from target 1 and 2.")
print("make poly Kernel Function")
accuracy(trained_breast_cancer_svm, X_test2, y_test2)
print("Changing C doesn't affect these classifications")

