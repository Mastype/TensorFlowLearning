# 使用Tensorboard 对 搭建的网络模型进行可视化

***
### 参考文档
*	[TensorBoard: Visualizing Learning](https://www.tensorflow.org/get_started/summaries_and_tensorboard#tensorboard-visualizing-learning)
*	[TensorBoard: Graph Visualization](https://www.tensorflow.org/get_started/graph_viz)
*	[TensorBoard's GitHub](https://github.com/tensorflow/tensorboard)
***

### 数据序列化

TensorBoard 是通过读取TensorFlow的事件文件来进行操作，其的大致生命周期如下：  
首先，在先创建需要搜集统计数据的TensorFlow，选择哪些节点需要进行汇总操作。(我理解使用Tensorflow为搭建模型结构），将需要记录的数据（summary operations）添加特定的标签（比如：'learning rate'，'loss function')。  
有关于汇总操作的详细信息在[*summary operations*](https://www.tensorflow.org/api_guides/python/summary)中

当Operations 在Tensorflow 中没有执行(run)或者有其他的op依赖于它的输出时，它是不会做任何事的。所以为了生存汇总信息，就需要执行所有的汇总节点。

然后进行合并操作（merged summary op)`merger = tf.summary.merge_all()`，将汇总的数据生存序列化的Summary protobuf对象。最终 将汇总数据记录在磁盘中，（即将protobuf对象传递给 tf.summary.FileWriter. FileWriter需要确定保存的位置，即声明-logdir 参数)

手写数据识别  
mnist with summaries [本地Code](../code/MINIST_Tensorboard.py) [官方GitHub](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py)

***
### Launch TensorBoard
执行  
`tensorboard --logdir=path/to/log-directory`  
或者是  
`python -m tensorboard.main`

`logdir` 指向FileWriter序列号存储的数据目录。然后在localhost:6006中查看TensorBoard。  
具体的TensorBoard 信息 请看 [**TensorBoard: Graph Visualization**](https://www.tensorflow.org/get_started/graph_viz)

***
### 实验
[**How To Use Tensorboard**](http://ischlag.github.io/2016/06/04/how-to-use-tensorboard/)

数据集使用的是MINIST数据


**写入记录文件及运行 Tensorboard**

未了能够记录日志文件首先需要一个SummaryWriter对象 声明如下

` writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())`

然后在训练过程中使用`Writer` 记录训练过程

`writer.add_summary(summary, global_step)`

在tensorboard 读取日志文件展示结果

`tensorboard --logdir=run1:/tmp/tensorflow/ --port 6006`

**定义tensor，绘制模型**

通过使用name 可以进行对绘制的图的tensor进行命名。为了让绘制的图具有较高的可读性，通过name_scope将节点整合在一起：
  
	with tf.name_scope('input'):
    	x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input") 
    	y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")  

绘制后的图形如下：

![architecture](\pic\tensorboard.png)

**记录动态变量**

为了验证模型在训练过程中的表现，就需要把，目标函数，预测精度进行输出绘制图标。   
创建相应的operations来记录这些变量的变化。

	tf.summary.scalar('cost', cross_entropy)
	tf.summary.scalar('accuracy', accuary)  
通过将operations 合并成单个(merged summary operation) 。省去多层置信所有operation的过程。  

	summary_op = tf.summary.merge_all()

**记录** 

最后通过在session 执行 train operation 和 summary operation。 在使用Summary Writer 将相应的信息记录在log目录下。

	# perform the operations we defined earlier on batch
	_, summary = sess.run([train_op, summary_op], feed_dict={x: batch_x, y_: batch_y})
	            
	# write log
	writer.add_summary(summary, epoch * batch_count + i)


**代码**  
[GitHub](https://github.com/Mastype/TensorFlowLearning/blob/master/code/Tensorboard.py)