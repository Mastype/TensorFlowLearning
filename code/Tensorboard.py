# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 21:33:24 2018
TensorBoard demo
目标是 将建立的模型进行可视化，以及对精度，目标函数(交叉熵)等训练过程的可视化
@author: dell
"""

import tensorflow as tf

# 用于在重新运行是重置
tf.reset_default_graph()

batch_size = 100
lr = 0.5
epochs = 100
log_path='/tmp/test'
if not tf.gfile.Exists(log_path):
    tf.gfile.MakeDirs(log_path)

data_path='/mnist_data'

# MINIST 手写数据， 数据是px 28*28 划分为0-9 10类
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(data_path, one_hot=True, source_url='http://yann.lecun.com/exdb/mnist/')

# 定义输入图
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 784], name='X-Input')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='Y-Input')

# 定义权值节点
with tf.name_scope('weights'):
    # 默认值为 0的 二维矩阵, 有将其设为服从标准分布初始值
    W = tf.Variable(tf.zeros([784, 10]))
    
# 定义偏移量
with tf.name_scope('biases'):
    # 默认值为 0 的 一维矩阵
    b = tf.Variable(tf.zeros([10]))

# 激活函数使用 逻辑回归
with tf.name_scope('activiction'):
    y = tf.nn.softmax(tf.matmul(x,W) + b)

# cost function
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y),reduction_indices=[1]))

# optimizer
with tf.name_scope('train'):
    # 随机梯度下降算法
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

# 预测精度
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 为目标函数和预测精度创建一个汇总值 create a summary
#tf.scalar_summay('cost', cross_entropy)  过期
#tf.scalar_summay('accuracy', accuary)
tf.summary.scalar('cost', cross_entropy)
tf.summary.scalar('accuracy', accuary)

# 将所有的汇总图和并为一个 operation 在 session 中执行
#summary_op = tf.merge_all_summaries()
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # 创建一个记录日志对象
#    writer = tf.train.SummaryWriter(log_path, graph=tf.get_default_graph())
    writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())
    
    for epoch in range(epochs):
        batch_count = int(mnist.train.num_examples/batch_size)
        
        for i in range(batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            
            # run
            _, summary = sess.run([train_op, summary_op], feed_dict={x: batch_x, y_: batch_y})
            
            # 每一代都将记录 可以稀疏一些
            writer.add_summary(summary, epoch * batch_count + i)
        
        if epoch % 5 == 0:
            print("Epoch: ", epoch)
    writer.close()
    print("精度: ", accuary.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))
    print("======END=========")