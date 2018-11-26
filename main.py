import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-1,1,100)
y = 2 * x + (np.random.random(100)*0.3)
x_train = tf.placeholder(dtype=tf.float32,shape=[1,1])
y_train = tf.placeholder(dtype=tf.float32,shape=[1,1])
W1 = tf.Variable(initial_value=tf.random_normal([1,10]),dtype=tf.float32,name = "W1")
B1 = tf.Variable(initial_value=tf.zeros([10]),name = "B1")
W2 = tf.Variable(initial_value=tf.random_normal([10,1]),dtype=tf.float32,name = "W2")
B2 = tf.Variable(initial_value=tf.zeros([1]),name = "B2")
layer_1 = tf.matmul(x_train,W1)+B1
layer_2 = tf.matmul(layer_1,W2)+B2
loss = tf.reduce_mean(tf.square(layer_2-y_train))
opti = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
init = tf.global_variables_initializer()
y_pre = []
saver = tf.train.Saver(max_to_keep=1)
config = tf.ConfigProto(log_device_placement = True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(init)
    for i in range(30):
        for j in range(100):
            sess.run(opti,feed_dict = {x_train:np.reshape(x[j],[1,1]),y_train:np.reshape(y[j],[1,1])})
        L = sess.run(loss,feed_dict = {x_train:np.reshape(x[50],[1,1]),y_train:np.reshape(y[50],[1,1])})
        print("第{0}次迭代的loss为{1}".format(i,L))
        saver.save(sess,"save/2x.ckpt-",global_step=i)
    
    for i in range(100):
        y_pre.append(sess.run(layer_2,feed_dict={x_train:np.reshape(x[i],[1,1])}))
y_pre = np.reshape(y_pre,[100])
print(y_pre.shape)
plt.figure(figsize=(8,8))
plt.plot(x,y,color = "red")
plt.plot(x,y_pre,color = "green")
plt.show()
