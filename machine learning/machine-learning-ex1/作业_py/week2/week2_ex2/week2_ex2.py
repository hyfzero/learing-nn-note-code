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
    plt.title("linear regression demo Multi-Variable")
    plt.xlabel("years")
    plt.ylabel("hosing price")
    plt.plot(X_raw, y_raw, "ob")
    plt.plot(x_dst, y_dst)
    plt.show()

data = pd.read_csv('ex1data2.txt', delimiter=',', names=['size', 'number of bedroom', 'price'], dtype=np.float32)
X1_raw = data['size'].values
X2_raw = data['number of bedroom'].values
y_raw = data['price'].values
X1 = (X1_raw-X1_raw.min())/(X1_raw.max()-X1_raw.min())
X2 = (X2_raw-X2_raw.min())/(X2_raw.max()-X2_raw.min())
y = (y_raw-y_raw.min())/(y_raw.max()-y_raw.min())

X1 = tf.constant(X1)
X2 = tf.constant(X2)
y = tf.constant(y)
theta1 = tf.Variable(initial_value=0., dtype=np.float32)
theta2 = tf.Variable(initial_value=0., dtype=np.float32)
theta3 = tf.Variable(initial_value=0., dtype=np.float32)
variables = [theta1, theta2, theta3]
num_epoch = 3000
j_history = {}
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)


for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = theta1 * X1 + theta2 * X2 + theta3
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
    # 自动计算损失函数关于自变量（模型参数）的梯度
    j_history[e] = loss
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
print(variables)
plot_cost_as_iteration(j_history)
y_dst = theta1 * X1 + theta2 * X2 + theta3
x_dst = X1*(X1_raw.max() - X1_raw.min())+X1_raw.min()
y_dst = y_dst*(y_raw.max() - y_raw.min())+y_raw.min()
plot_linear_regression(X1_raw, y_raw, x_dst, y_dst)