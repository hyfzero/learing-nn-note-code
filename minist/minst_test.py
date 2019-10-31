from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# MNIST数据集相关的常数。
INPUT_NODE = 784    # 输入层的节点数。对于MNIST数据集，这个就是图片像素
OUTPUT_NODE = 10    #输出层的节点数。0-9这十个数


# 配置神经网络相关的常数
LAYER1_NODE = 500  # 隐藏层节点数。这里使用一个隐藏层，这个隐藏层有500个节点
BATCH_SIZE = 100   # 一个训练batch中的训练数据个数
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99    # 学习的衰减率
REGULARIZATION_RATE = 0.0001
TRAINING_STEP = 30000
MOVING_AVERAGE_DECAY = 0.99


# 一个辅助函数，给定神经网络的输入和所有参数，计算前向传播网络。Relu激活函数三层神经网络加入了计算平均值的类
def inference(input_tensor, avg_class, weight1, biases1, weight2, biases2):
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1)+biases1)
        return tf.matmul(layer1, weight2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weight1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weight2)) + avg_class.average(biases2)




def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

    weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    # 不使用滑动平均
    y = inference(x, None, weight1, biases1, weight2, biases2)

    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。在第4章中介绍过给
    # 定训练轮数的变量可以加快训练早期变量的更新速度。
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )

    # 在所有代表神经网络参数的变量上使用滑动平均。其他辅助变量(比如global_step)就
    # 不需要了。tf.trainable_variable返回的就是图上集合
    # GraphKeys.TRAINABLE_VARIABLES中的元素。这个集合的元素就是所有没有指定
    # trainable=False的参数。
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables()
    )

    # 注意这个与上面的y有什么区别。计算使用了滑动平均之后的前向传播结果。第4章中介绍过滑动平均不会改变
    # 变量本身的取值，而是会维护一个影子变量来记录其滑动平均值。所以当需要使用这个滑动平均值时，
    # 需要明确调用average函数。
    average_y = inference(
        x, variable_averages, weight1, biases1, weight2, biases2
    )

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
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失。一般只计算神经网络边上权重的正则化损失，而不使用偏置项。
    regularization = regularizer(weight1) + regularizer(weight2)
    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
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
        train_op = tf.no_op(name='train')  # tf.no_op是一个没有实际意义的函数

    # 检验使用了滑动平均模型的神经网络前向传播结果是否正确。tf.argmax(average_y, 1)
    # 计算每一个样例的预测结果。其中average_y是一个batch_size * 10的二维数组，每一行
    # 表示一个样例的前向传播结果。tf.argmax的第二个参数“1”表示选取最大值的操作仅在第一
    # 个维度中进行，也就是说，只在每一行选取最大值对应的下标。于是得到的结果是一个长度为
    # batch的一维数组，这个一维数组中的值就表示了每一个样例对应的数字识别结果。tf.equal
    # 判断两个张量的每一维是否相等，如果相等返回True，否则返回False。
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 注意这个accuracy是只跟average_y有关的，跟y是无关的
    # 这个运算首先讲一个布尔型的数值转化为实数型，然后计算平均值。这个平均值就是模型在这
    # 一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 前面的所有步骤都是在构建模型，将一个完整的计算图补充完了，现在开始运行模型
    # 初始化会话并且开始训练过程
    with tf.Session() as sess:
        # 初始化变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 准备验证数据。一般在神经网络的训练过程中会通过验证数据要大致判断停止的
        # 条件和评判训练的效果。
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }
        # 准备测试数据。在真实的应用中，这部分数据在训练时是不可见的，这个数据只是作为
        # 模型优劣的最后评价标准。
        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
        }
        # 认真体会这个过程，整个模型的执行流程与逻辑都在这一段
        # 迭代的训练神经网络
        for i in range(TRAINING_STEP):
            # 每1000轮输出一次在验证数据集上的测试结果
            if i % 1000 == 0:
                # 计算滑动平均模型在验证数据上的结果。因为MNIST数据集比较小，所以一次
                # 可以处理所有的验证数据。为了计算方便，本样例程序没有将验证数据划分为更
                # 小的batch。当神经网络模型比较复杂或者验证数据比较大时，太大的batch
                # 会导致计算时间过长甚至发生内存溢出的错误。
                # 注意我们用的是滑动平均之后的模型来跑我们验证集的accuracy
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy "
                      "using average model is %g " % (i, validate_acc))

            # 产生这一轮使用的一个batch的训练数据，并运行训练过程。
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 在训练结束之后，在测试数据上检测神经网络模型的最终正确率。
        # 同样，我们最终的模型用的是滑动平均之后的模型，从这个accuracy函数
        # 的调用就可以看出来了，因为accuracy只与average_y有关
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average "
              "model is %g" % (TRAINING_STEP, test_acc))


# 主程序入口

# 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据。
mnist = input_data.read_data_sets("minst_data", one_hot=True)
print("Training data 0 label:", mnist.train.labels[0].shape)
train(mnist)



