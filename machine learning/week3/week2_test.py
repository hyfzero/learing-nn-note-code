import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def maxminnorm1(array):
    maxcols=array.max(axis=0)
    mincols=array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in xrange(data_cols):
        t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
    return t

# Importing data
data = pd.read_csv('ex2data1.txt', delimiter=',', names=['Exam1 Score', 'Exam2 Score', 'Admitted'])
X = data[['Exam1 Score', 'Exam2 Score']].values
y = data['Admitted'].values

m, n = np.shape(X)
y = y.reshape(m, 1)
print(X, y)
print(X[y[:, 0] == 0, 0])

# Plot the data
def plot_raw_data(X, y, show = False):
    admitted = plt.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], marker='+', color='b')
    not_admitted = plt.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], marker='o', color='y')
    plt.xlabel('Exam1 Score')
    plt.ylabel('Exam2 Score')
    plt.legend((admitted, not_admitted), ("Admitted", "Not admitted"))
    print(X)
    if show is True:
        plt.show()


plot_raw_data(X, y, True)


# Define the cost function and gradient
def sigmoid(X):
    return 1/(1 + np.exp(-X))


def cost_function(theta, X, y):
    m, n = np.shape(X)
    theta = theta.reshape(n , 1)
    h = sigmoid(X.dot(theta))
    j = (1/m)*(-y.T.dot(np.log(h)) - (1-y).T.dot(np.log(1-h)))
    grad = (1/m)*X.T.dot(h-y)
    return j, grad


# Test the accuracy of cost function
X = np.append(np.ones((m, 1)), X, axis = 1)
initial_theta = np.zeros(((n+1), 1))
print('The cost should be around 0.693')
print(cost_function(initial_theta, X, y)[0])

# Fit parameter theta using truncated Newton algorithm
initial_theta = np.zeros(((n+1))) # initial  theta must be 1 d array
result = opt.fmin_tnc(func = cost_function, x0=initial_theta, args=(X, y))
optimal_theta = result[0].reshape(3,1)


# Plot decision boundary
def plot_decision_boundary(X, y, theta):
    plot_raw_data(X[:,1:], y, False)
    exam1 = np.linspace(30, 100, 5)
    exam2 = -(theta[0] + theta[1]*exam1)/ theta[2]
    plt.plot(exam1,exam2)
    plt.show()


plot_decision_boundary(X, y, optimal_theta)


# Predict the admission chance based on score
score = np.array([1, 45, 85]).reshape(1,3)
admission_probability = sigmoid(score.dot(optimal_theta))
print('The probability of being admitted with Exam1 score: 45 and Exam2 score:85 should be around 0.78')
print(admission_probability)