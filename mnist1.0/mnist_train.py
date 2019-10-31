import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
import mnist_inference

# 训练参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "E:\日常\学习\deep learning\mnist1.0\mnist_saver"
MODEL_NAME = "model.ckpt"


def train(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input")
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 定义前向传播
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。在第4章中介绍过给
    # 定训练轮数的变量可以加快训练早期变量的更新速度。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数。这里使用了TensorFlow中提
    # 供的sparse_softmax_cross_entropy_with_logits函数来计算交叉熵。当分类
    # 问题只有一个正确答案时，可以使用这个函数来加速交叉熵的计算。MNIST问题的图片中
    # 只包含了0~9中的一个数字，所以可以使用这个函数来计算交叉熵损失。这个函数的第一个
    # 参数是神经网络不包括softmax层的前向传播结果，第二个是训练数据的正确答案。因为
    # 标准答案是一个长度位10的一维数组，而该函数需要提供的是一个正确答案的数字，所以需
    # 要使用tf.argmax函数来得到正确答案对应的类别编号。
    # 注意这里用的是y来计算交叉熵而不是average_y
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础的学习率，随着迭代的进行，更新变量时使用的
        # 学习率在这个基础上递减
        global_step,  # 当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,  # 过完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY  # 学习率的衰减速度
    )
    # 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数。注意这里损失函数
    # 包含了交叉熵损失和L2正则化损失。
    # 在这个函数中，每次执行global_step都会加一。注意这个函数优化的损失函数跟y有关，
    # 跟average_y无关。
    train_step = tf.train.GradientDescentOptimizer(learning_rate) \
        .minimize(loss, global_step=global_step)

    # 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，
    # 又要更新每个参数的滑动平均值。为了一次完成多个操作，TensorFlow提供了
    # tf.control_dependencies和tf.group两种机制。下面两行程序和
    # train_op = tf.group(train_step, variables_average_op)是等价的。
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step],feed_dict={x: xs, y_: ys})
            if i % 1000 is 0:
                print("After %d training step(s), validation accuracy "
                  "using average model is %g " % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


# 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据。
mnist = input_data.read_data_sets("mnist_train_example", one_hot=True)
print("Training data 0 label:", mnist.train.labels[0].shape)
train(mnist)



