import tensorflow as tf
import numpy as np
sess = tf.Session()

#define parameters
M1 = 8
M2 = 3
N = 1
k = 4
responsibility = 1/float(k)
X = tf.placeholder(tf.float32, [M1,N])
Z = tf.placeholder(tf.float32, [M2,N])
ArrayX = np.random.rand(M1, N)
ArrayY = np.random.rand(M2, N)

#calculate pairwise distance
D = tf.reduce_sum(tf.square(X),1)
E = tf.reduce_sum(tf.square(Z),1)
Square_X = tf.reduce_sum(tf.square(X),1)
Square_Z = tf.reduce_sum(tf.square(Z),1)
InnerProduct = tf.matmul(X, tf.transpose(Z))
res = tf.transpose(Square_X + tf.transpose(Square_Z - 2*InnerProduct))
print (sess.run(res, feed_dict={X:ArrayX, Z:ArrayY}))

#assign responsibility
topK = tf.nn.top_k(-tf.transpose(res), k)
b1 = topK.indices
b = sess.run(b1, feed_dict={X:ArrayX, Z:ArrayY})
index_array = np.linspace(0, M2-1, M2,dtype = int)
index_array = index_array.repeat(k)
b = b.reshape(M2*k,)
result = np.zeros([M2,M1])
result[index_array,b] = responsibility
print (np.transpose(result))



