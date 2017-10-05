import tensorflow as tf
import numpy as np
sess = tf.Session()
N = 2
B = 3
C = 6
X = tf.placeholder(tf.float32, [B,N])
Z = tf.placeholder(tf.float32, [C,N])
RandArrayX = np.random.rand(B,N)
RandArrayZ = np.random.rand(C,N)
Square_X = tf.reduce_sum(tf.square(X),1)
Square_Z = tf.reduce_sum(tf.square(Z),1)
InnerProduct = tf.matmul(X, tf.transpose(Z))
res = tf.transpose(Square_X + tf.transpose(Square_Z - 2*InnerProduct))
print (sess.run(X*1, feed_dict={X:RandArrayX}))
print (sess.run(Z*1, feed_dict={Z:RandArrayZ}))
print (sess.run(res, feed_dict={X:RandArrayX, Z:RandArrayZ}))




