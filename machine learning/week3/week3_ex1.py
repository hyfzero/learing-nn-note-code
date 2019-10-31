import tensorflow as tf
from numpy.random import RandomState
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# 归一化
def normalize(data2):
    maxcols = data2.max(axis=0)
    mincols = data2.min(axis=0)
    data_shape = data2.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t = np.empty((data_rows, data_cols))
    for i in range(data_cols):
        t[:, i] = (data2[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
    return t

def plot_raw_data(X, y, show = False):
    admitted = plt.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], marker='+', color='b')
    not_admitted = plt.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], marker='o', color='y')
    plt.xlabel('Exam1 Score')
    plt.ylabel('Exam2 Score')
    plt.legend((admitted, not_admitted), ("Admitted", "Not admitted"))
    if show is True:
        plt.show()


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
a = tf.matmul(x, w1) + biases1
y1 = tf.matmul(a, w2) + biases2
# 定义损失函数和反向传播算法
y = tf.sigmoid(y1)
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))+(1-y_) * tf.log(tf.clip_by_value(
     1-y, 1e-10, 1.0)))
train_step = tf.train.AdagradOptimizer(0.09).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
# Importing data
data = pd.read_csv('ex2data1.txt', delimiter=',', names=['Exam1 Score', 'Exam2 Score', 'Admitted'])
X = data[['Exam1 Score', 'Exam2 Score']].values
Y = data['Admitted'].values

m, n = np.shape(X)
X = normalize(X)
Y = Y.reshape(m, 1)
dataset_size = 100

print(X, Y)
plot_raw_data(X, Y, True)
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(biases1))
    print(sess.run(biases2))
    print(sess.run(w1))
    print(sess.run(w2))

    # w1 = [[-0.81131822, 1.4859876, 0.06532937], [-2.44270396, 0.0992484, 0.59122431]]
    # w2 = [[-0.81131822], [1.4859876], [0.06532937]]

    STEPS = 10000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entry = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("%d steps,w is %g" % (i, total_cross_entry))
    print(sess.run(w1))
    print(sess.run(w2))
    exam2 = sess.run(y, feed_dict={x: X})
    ex = np.concatenate((Y, exam2), axis=1).round(0)
    print(ex)


