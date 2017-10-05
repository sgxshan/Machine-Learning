import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
sess = tf.Session()
def reduce_logsumexp(input_tensor, reduction_indices=1, keep_dims=False):
    max_input_tensor1 = tf.reduce_max(input_tensor,reduction_indices, keep_dims=keep_dims)
    max_input_tensor2 = max_input_tensor1
    if not keep_dims:
        max_input_tensor2 = tf.expand_dims(max_input_tensor2,reduction_indices)
    return tf.log(tf.reduce_sum(tf.exp(input_tensor - max_input_tensor2),reduction_indices, keep_dims=keep_dims)) + max_input_tensor1

def logsoftmax(input_tensor):
    return input_tensor - reduce_logsumexp(input_tensor, keep_dims=True)

#calculate pairwise distance
def pairwise_distance(X,Z):
    Square_X = tf.reduce_sum(tf.square(X),1)
    Square_Z = tf.reduce_sum(tf.square(Z),1)
    InnerProduct = tf.matmul(X, tf.transpose(Z))
    res = tf.transpose(Square_X + tf.transpose(Square_Z - 2*InnerProduct))
    return res

#declear a function to minimize total loss
def buildGraph(learning_rate,K):
    D = 2
    U = tf.Variable(tf.random_normal(shape=[K, 2], stddev=0.5))
    FI = tf.Variable(tf.random_normal(shape=[1,K], stddev=0.5))
    std_fi = tf.Variable(tf.random_normal(shape=[1,K], stddev=0.5))
    X = tf.placeholder(tf.float32, [None, 2])
    distance = pairwise_distance(X,U)
    log_prior = logsoftmax(FI)
    log_likelihood = -D/2 * tf.log(2.*math.pi*tf.exp(std_fi)) - tf.div(distance,(2.*tf.exp(std_fi)))
    total_loss  = -1 * tf.reduce_sum(reduce_logsumexp(log_prior + log_likelihood, 1, True))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.99,epsilon=1e-5)
    train = optimizer.minimize(loss=total_loss)
    return train, X, U, total_loss


#declear a function to display training process
def runGraph(learning_rate,K):
    train, X, U, Loss= buildGraph(learning_rate,K)
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)
    loss_list = []
    for step in range(0,1000):
        loss,train_process= sess.run([Loss, train], feed_dict={X:np.load("data2D.npy")})
        loss_list.append(loss)
    return loss_list

learning_rate = 0.05
total_loss = runGraph(learning_rate, 3)
print (total_loss)
plt.plot(total_loss, label='learning rate = 0.001')
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