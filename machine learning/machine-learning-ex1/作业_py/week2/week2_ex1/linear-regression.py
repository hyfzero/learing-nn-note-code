import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
tf.enable_eager_execution(
    config=None,
    device_policy=None,
    execution_mode=None
)
# Plot the cost function with iteration
def plot_cost_as_iteration ( j_history ):
    lists = sorted(j_history.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y, marker='o', markevery=50)
    plt.title('Cost History')
    plt.xlabel('Number of iteration')
    plt.ylabel('Cost')
    plt.show()
# Plot the result of linear regression
def plot_linear_regression(X_raw, y_raw,x_dst, y_dst):
    plt.title("linear regression demo")
    plt.xlabel("years")
    plt.ylabel("hosing price")
    plt.plot(X_raw, y_raw, "ob")
    plt.plot(x_dst, y_dst)
    plt.show()


data = pd.read_csv('ex1data1.txt', sep=',', names=['population', 'profit'], dtype=np.float32)
X_raw = data['population'].values
y_raw = data['profit'].values

# 归一化
X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())




X = tf.constant(X)
y = tf.constant(y)
a = tf.Variable(initial_value=0. , dtype=np.float32)
b = tf.Variable(initial_value=0. , dtype=np.float32)
variables = [a, b]
num_epoch = 3000
j_history = {}
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
    # 自动计算损失函数关于自变量（模型参数）的梯度
    j_history[e] = loss
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

print(variables)
plot_cost_as_iteration(j_history)
y_dst = a * X + b
x_dst = X*(X_raw.max() - X_raw.min())+X_raw.min()
y_dst = y_dst*(y_raw.max() - y_raw.min())+y_raw.min()
plot_linear_regression(X_raw, y_raw,x_dst, y_dst)