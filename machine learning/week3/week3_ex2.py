import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# Plotting the data
def plot_raw_data(X, y, show = False):
    passed = plt.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], marker='+', color='b')
    not_passed = plt.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], marker='o', color='y')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend((passed, not_passed), ("Pass tests", "Fail to pass tests"))
    if show is True:
        plt.show()


# Importing data
data = pd.read_csv('ex2data2.txt', delimiter=',', names=['Microchip Test 1', 'Microchip Test 2', 'Passed'])
X = data[['Microchip Test 1',
          'Microchip Test 2']].values
Y = data['Passed'].values

m, n = np.shape(X)
Y = Y.reshape(m, 1)

print(X, Y)
plot_raw_data(X, Y, True)


# 定义训练集的大小
batch_size = 5
biases1 = tf.Variable(tf.random_normal([1, 3], stddev=1, seed=1))
biases2 = tf.Variable(tf.random_normal([1, 1], stddev=1, seed=1))
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 神经网络的参数
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
# 定义神经网络前向传播
a = tf.nn.relu(tf.matmul(x, w1) + biases1)
y1 = tf.nn.relu(tf.matmul(a, w2) + biases2)
# 定义损失函数和反向传播算法
y = y1
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))+(1-y_) * tf.log(tf.clip_by_value(
     1-y, 1e-10, 1.0))) + tf.contrib.layers.l2_regularizer(0.01)(w1)+tf.contrib.layers.l2_regularizer(0.01)(w2)
train_step = tf.train.AdagradOptimizer(0.09).minimize(cross_entropy)
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(biases1))
    print(sess.run(biases2))
    print(sess.run(w1))
    print(sess.run(w2))
    STEPS = 5000
    for i in range(STEPS):
        sess.run(train_step, feed_dict={x: X, y_: Y})
        if i % 1000 == 0:
            total_cross_entry = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("%d steps,w is %g" % (i, total_cross_entry))
    print(sess.run(w1))
    print(sess.run(w2))
    exam2 = sess.run(y, feed_dict={x: X}).round(0)
    print(exam2.round(0))
    ex = np.concatenate((Y, exam2), axis=1).round(0)
    passed = plt.scatter(X[[exam2[:, 0] == 1, 0]], X[exam2[:, 0] == 1, 1], marker='+', color='b')
    not_passed = plt.scatter(X[exam2[:, 0] == 0, 0], X[exam2[:, 0] == 0, 1], marker='o', color='y')
    plt.xlabel('Microchip Test ')
    plt.ylabel('Microchip Test ')
    plt.legend((passed, not_passed), ("Pass tests", "Fail to pass tests"))
    plt.show()
    print(ex)
