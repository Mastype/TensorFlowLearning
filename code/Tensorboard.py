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
epochs = 1000
hidden_size = 10
log_path='test'
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



# 激活函数使用 逻辑回归
# 定义权值节点
with tf.name_scope('weights'):
# 默认值为 0的 二维矩阵, 有将其设为服从标准分布初始值
    W = tf.Variable(tf.zeros([784, 10]))
    
    # 定义偏移量
with tf.name_scope('biases'):
        # 默认值为 0 的 一维矩阵
    b = tf.Variable(tf.zeros([10]))

with tf.name_scope('output'): 
    y = tf.nn.softmax(tf.matmul(x,W) + b)

    
# cost function
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))

# optimizer
with tf.name_scope('train'):
    # 随机梯度下降算法
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

# 预测精度
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 为目标函数和预测精度创建一个汇总值 create a summary
#tf.scalar_summay('cost', cross_entropy)  过期
#tf.scalar_summay('accuracy', accuary)
tf.summary.scalar('cost', cross_entropy)
tf.summary.scalar('accuracy', accuracy)

# 将所有的汇总图和并为一个 operation 在 session 中执行
#summary_op = tf.merge_all_summaries()
summary_op = tf.summary.merge_all()

# 创建一个记录日志对象
#writer = tf.train.SummaryWriter(log_path, graph=tf.get_default_graph())
writer = tf.summary.FileWriter(log_path)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())    
    
    for epoch in range(epochs):
        batch_count = int(mnist.train.num_examples/batch_size)
        
#        for i in range(batch_count):
#            batch_x, batch_y = mnist.train.next_batch(batch_size)
#            
#            # perform the operations we defined earlier on batch
#            _, summary,acc = sess.run([train_op, summary_op, accuracy], feed_dict={x: batch_x, y_: batch_y})
#            
#            # write log
#            writer.add_summary(summary, epoch * batch_count + i)
        if epoch % 10 == 0:
            # write log
            test_summary,acc = sess.run([summary_op, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            writer.add_summary(test_summary, epoch)
            print("Epoch: %d . acc: %2f" %(epoch, acc))
        else:            
            # perform the operations we defined earlier on batch
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, acc = sess.run([train_op, accuracy], feed_dict={x: batch_x, y_: batch_y})
            
    writer.close()
    print("精度: ", accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))
    print("======END=========")