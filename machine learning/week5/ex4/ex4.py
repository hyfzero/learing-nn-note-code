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

# Set up shared parameters for NN
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
lambda_reg = 1


# Define the function to display data
def getDatumImg(row):

    """
    Function that is handed a single np array with shape 1x400,
    crates an image object from it, and returns it
    """
    width, height = 20, 20
    square = row[1:].reshape(width, height)
    return square.T


def displayData(x):
    """
    Function that picks 100 random rows from X, creates a 20x20 image from each,
    then stitches them together into a 10x10 grid of images, and shows it.
    """
    width, height = 20, 20
    nrows, ncols = 10, 10

    # if is used to visualize hidden layer
    if x.shape[0] < nrows * ncols:
        nrows, ncols = 5, 5

    indices_to_display = random.sample(range(0, x.shape[0]), nrows * ncols)

    big_picture = np.zeros((height * nrows, width * ncols))

    irow, icol = 0, 0

    for idx in indices_to_display:
        if icol == ncols:
            irow += 1
            icol = 0

        iimg = getDatumImg(x[idx])
        big_picture[irow * height:irow * height + iimg.shape[0], icol * width:icol * width + iimg.shape[1]] = iimg

        icol += 1

    plt.imshow(big_picture, cmap=cm.Greys_r)
    plt.axis('off')
    plt.show()


displayData(X)


# Define the cost function
def sigmoid(x):
    return 1/(1 + np.exp(-x))


def nn_cost_function(theta, num_labels, X, y, lambda_reg):
    m = X.shape[0]

    # Unroll 1d parameter
    theta_input_to_hidden = theta[0: hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, input_layer_size + 1))
    theta_hidden_to_output = theta[hidden_layer_size * (input_layer_size + 1): ].reshape((num_labels, hidden_layer_size + 1))

    # Forwardfeed to calculate cost
    z2 = X.dot(theta_input_to_hidden.T)
    a2 = sigmoid(z2)
    a2 = np.append(np.ones((m, 1)), a2, axis=1)
    z3 = a2.dot(theta_hidden_to_output.T)
    h = sigmoid(z3)

    y_multi = np.zeros((m, num_labels))
    for i in range(m):
        y_multi[i, y[i] - 1] = 1

    penalty_input_layer = np.sum(np.sum(np.square(theta_input_to_hidden[:, 1:]), axis=1))
    penalty_hidden_layer = np.sum(np.sum(np.square(theta_hidden_to_output[:, 1:]), axis=1))
    penalty = lambda_reg * (penalty_input_layer + penalty_hidden_layer) / (2*m)

    j = -1 / m * ((np.sum(np.sum(np.log(h) * y_multi, axis=1))) + np.sum(np.sum(np.log(1-h) * (1 - y_multi), axis=1))) + penalty

    # Backpropagation to compute gradient
    sigma3 = h - y_multi
    sigma2 = sigma3.dot(theta_hidden_to_output) * a2 * (1 - a2)
    sigma2 = sigma2[:, 1:]

    delta2 = sigma3.T.dot(a2)
    delta1 = sigma2.T.dot(X)

    p2 = lambda_reg / m * (np.c_[np.zeros((theta_hidden_to_output.shape[0], 1)), theta_hidden_to_output[:, 1:]])
    p1 = lambda_reg / m * (np.c_[np.zeros((theta_input_to_hidden.shape[0], 1)), theta_input_to_hidden[:, 1:]])
    grad = np.concatenate(((delta1 / m + p1).flatten(), (delta2 / m + p2).flatten()))

    return j, grad.flatten()


# Test the accuracy of cost function
weights = scipy.io.loadmat('ex4weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']
theta = np.concatenate((theta1.flatten(), theta2.flatten()))
cost, gradient = nn_cost_function(theta, num_labels, X, y, lambda_reg)
print('The cost should be around: 0.383770')
print(cost)


# Random Initialization
def random_initialize(m, n):
    epsilon_init = 0.12
    return np.random.rand(m, n) * 2 * epsilon_init - epsilon_init


initialization_theta1 = random_initialize(hidden_layer_size, input_layer_size + 1)
initialization_theta2 = random_initialize(num_labels, hidden_layer_size + 1)
initialization_theta = np.concatenate((initialization_theta1.flatten(), initialization_theta2.flatten()))
cost, gradient = nn_cost_function(initialization_theta, num_labels, X, y, lambda_reg)
print(cost)
# Predict the function to predict results
def predict_nn(theta, X):
    theta_input_to_hidden = theta[0: hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, input_layer_size + 1))
    theta_hidden_to_output = theta[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, hidden_layer_size + 1))

    hidden_layer = sigmoid(X.dot(theta_input_to_hidden.T))
    m, n = hidden_layer.shape
    hidden_layer = np.append(np.ones((m, 1)), hidden_layer, axis=1)
    output_layer = sigmoid(hidden_layer.dot(theta_hidden_to_output.T))
    predict = output_layer.argmax(axis=1) + 1
    return predict


# Fit the theta (initialization_theta must be 1d)
optimal_theta = opt.minimize(fun=nn_cost_function, method='CG', jac=True, x0=initialization_theta.flatten(), args=(num_labels, X, y, lambda_reg),
                          options={'maxiter': 300, 'disp': False}).x

predict_nn = predict_nn(optimal_theta, X)
print('\nTraining Set Accuracy when using neural network with 300 iteration is: ')
print(np.mean((predict_nn == y.flatten()).astype(int))*100)


# Visualize the hidden layer
displayData(optimal_theta[0: hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1))