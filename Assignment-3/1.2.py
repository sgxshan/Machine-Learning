import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
sess = tf.Session()

#calculate pairwise distance
def pairwise_distance(X,Z):
    Square_X = tf.reduce_sum(tf.square(X),1)
    Square_Z = tf.reduce_sum(tf.square(Z),1)
    InnerProduct = tf.matmul(X, tf.transpose(Z))
    res = tf.transpose(Square_X + tf.transpose(Square_Z - 2*InnerProduct))
    return res

#declear a function to minimize total loss
def buildGraph(learning_rate,K):
    U = tf.Variable(tf.random_normal(shape=[K, 2], stddev=0.5))
    X = tf.placeholder(tf.float32, [None, 2])
    distance = pairwise_distance(X,U)
    index = tf.argmin(distance,1)
    cluster = tf.gather(U,index)
    square_loss = tf.reduce_sum(tf.reduce_sum(tf.square(X - cluster),1),0)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.99,epsilon=1e-5)
    train = optimizer.minimize(loss=square_loss)
    return train, X, U, square_loss

#declear a function to display training process
def runGraph(learning_rate,K):
    train, X, U, square_loss = buildGraph(learning_rate,K)
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)
    loss_list = []
    for step in range(0,250):
        u,train_process,loss= sess.run([U, train,square_loss], feed_dict={X:np.load("data2D.npy")})
        loss_list.append(loss)
    return loss_list


learning_rate = 0.001

total_loss = runGraph(learning_rate,3)

print (total_loss)
plt.plot(total_loss,label = 'learning rate = 0.001')
plt.show()
plt.title('Total Entropy Loss Versus Updating Numbers (minBatchSize = 500)')
plt.xlabel('Number of Updates')
plt.ylabel('Total loss')



# a=tf.constant([1,2,3,4,5,6,7,8,11,12,9,10])
#
# a = tf.reshape(a,[6,2])
#
# b=tf.constant([9,10,11,12,1,1])
# b=tf.reshape(b,[3,2])
# d = pairwise_distance(a,b)
# c=tf.argmin(d,1)
#
#
# e=tf.gather(b,c)
# print sess.run(a)
# print sess.run(b)
# print sess.run(d)
# print sess.run(c)
# print sess.run(e)