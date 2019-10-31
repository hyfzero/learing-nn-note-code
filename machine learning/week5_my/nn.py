import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.io
import random
import matplotlib.cm as cm

# Importing data
data = scipy.io.loadmat('ex4data1.mat')
X, y = data['X'], data['y']
m, n = X.shape
X = np.append(np.ones((m, 1)), X, axis=1)

weights = scipy.io.loadmat('ex4weights.mat')
# Set up shared parameters for NN
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
lambda_reg = 1
learning_rate = 0.05
training_steps = 500000


# Random Initialization
def random_initialize(m, n):
    epsilon_init = 0.12
    return np.random.rand(m, n) * 2 * epsilon_init - epsilon_init


# initialization_theta1 = random_initialize(input_layer_size+1, hidden_layer_size)
# initialization_theta2 = random_initialize(hidden_layer_size+1, num_labels)
initialization_theta1 = np.load( "theta1.npy" )
initialization_theta2 = np.load( "theta2.npy" )



# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))


def forward_propagation(x, theta1, theta2):
    hidden_layer1 = sigmoid(x.dot(theta1))
    m, n = hidden_layer1.shape
    hidden_layer1 = np.append(np.ones((m, 1)), hidden_layer1, axis=1)
    output = sigmoid(hidden_layer1.dot(theta2))
    return output


def h1(x, theta1):
    hidden_layer1 = sigmoid(x.dot(theta1))
    m, n = hidden_layer1.shape
    hidden_layer1 = np.append(np.ones((m, 1)), hidden_layer1, axis=1)
    return hidden_layer1


y_ = forward_propagation(X, initialization_theta1, initialization_theta2)


def nn_cost_function(theta1, theta2, num_label, X, y, lambda_reg):
    m = X.shape[0]
    h = forward_propagation(X, theta1, theta2)
    y_multi = np.zeros((m, num_label))
    for i in range(m):
        y_multi[i, y[i] - 1] = 1
    penalty_input_layer = np.sum(np.sum(np.square(theta1[:, 1:]), axis=1))
    penalty_hidden_layer = np.sum(np.sum(np.square(theta2[:, 1:]), axis=1))
    penalty = lambda_reg * (penalty_input_layer + penalty_hidden_layer) / (2 * m)
    j = -1 / m * ((np.sum(np.sum(np.log(h) * y_multi, axis=1))) + np.sum(
        np.sum(np.log(1 - h) * (1 - y_multi), axis=1))) + penalty
    return j


cost = nn_cost_function(initialization_theta1, initialization_theta2, num_labels, X, y, lambda_reg)
print("cost is " + str(cost))
# 反向传播算法
def nn_training(theta1, theta2, num_label, X, y, lambda_reg, training_steps, learning_rate):
    m = X.shape[0]
    y_multi = np.zeros((m, num_label))
    for i in range(m):
        y_multi[i, y[i]-1] = 1
    for i in range(training_steps):
        sample_num = i % 5000
        h2 = forward_propagation(X, theta1, theta2)
        duty1 = h2 * (1 - h2) * (y_multi - h2)
        delta = theta2.dot(duty1[sample_num])
        h_1 = h1(X, theta1)
        duty2 = (h_1[sample_num] * (1 - h_1[sample_num]) * delta)[1:26]
        theta2 = theta2 + learning_rate * (duty1[sample_num].reshape(duty1[sample_num].shape[0],
                                    1)).dot(h_1[sample_num].reshape(1, h_1[sample_num].shape[0])).T
        theta1 = theta1 + learning_rate * (duty2.reshape(duty2.shape[0],
                                    1)).dot(X[sample_num].reshape(1, X[sample_num].shape[0])).T
        if i % 1000 is 0:
            np.save("theta1.npy", theta1)
            np.save("theta2.npy", theta2)
            cost1 = nn_cost_function(theta1, theta2, num_labels, X, y, lambda_reg)
            print("after " + str(i) + " steps, cost is " + str(cost1))
    cost1 = nn_cost_function(theta1, theta2, num_labels, X, y, lambda_reg)
    print(cost1)


def rate_same(y1, y2):
    m = y1.shape[0]
    same = 0
    for i in range(m):
        if y1[i] == y2[i]:
            same = same + 1
    return same/m


y_ = forward_propagation(X, initialization_theta1, initialization_theta2)
y_ = y_.argmax(axis=1).reshape(y_.shape[0], 1)+1
rate = rate_same(y, y_)
print("correcting rate is " + str(rate*100) + "%.")
#nn_training(initialization_theta1, initialization_theta2, num_labels, X, y, lambda_reg, training_steps, learning_rate)