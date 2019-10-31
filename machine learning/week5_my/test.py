import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.io
import random
import matplotlib.cm as cm

# Importing data
data = scipy.io.loadmat('ex4data1.mat')
X = np.array([0.8, 0.1])
m = X.shape

weights = scipy.io.loadmat('ex4weights.mat')
# Set up shared parameters for NN
input_layer_size = 2
hidden_layer_size = 2
num_labels = 2
lambda_reg = 1


# Random Initialization
def random_initialize(m, n):
    epsilon_init = 0.12
    return np.random.rand(m, n) * 2 * epsilon_init - epsilon_init


initialization_theta1 = np.array([[-1, 0.1], [0.5, 0.7]])
initialization_theta2 = np.array([[0.9, 0.5], [0.3, 0.1]])
print(initialization_theta1.shape)

# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))


def forward_propagation(x, theta1, theta2):
    hidden_layer1 = sigmoid(x.dot(theta1))
    print(hidden_layer1)
    output = sigmoid(hidden_layer1.dot(theta2))
    return output


y_ = forward_propagation(X, initialization_theta1, initialization_theta2)


