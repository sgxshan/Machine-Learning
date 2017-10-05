import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
sess = tf.Session()
np.random.seed(521)
data = np.load("data100D.npy")
print np.shape(data)
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

def log_prob_density(X,U,std_fi,pi,D):
    distance = pairwise_distance(X,U)
    log_likelihood = -D/2 * tf.log(2.*math.pi*tf.exp(std_fi)) - tf.div(distance,(2.*tf.exp(std_fi)))
    log_prior = logsoftmax(pi)
    log_magrinal = tf.reduce_sum(reduce_logsumexp(log_prior + log_likelihood, 1, True))
    log_posterior = log_likelihood + log_prior - log_magrinal
    return log_posterior

def loss_calculation(X,U,std_fi,FI,D):
    distance = pairwise_distance(X,U)
    log_prior = logsoftmax(FI)
    log_likelihood = -D/2 * tf.log(2.*math.pi*tf.exp(std_fi)) - tf.div(distance,(2.*tf.exp(std_fi)))
    total_loss  = -1 * tf.reduce_sum(reduce_logsumexp(log_prior + log_likelihood, 1, True))
    return total_loss

#declear a function to minimize total loss
def buildGraph(learning_rate,K, D):
    U = tf.Variable(tf.random_normal(shape=[K, 100], stddev=0.5))
    FI = tf.Variable(tf.random_normal(shape=[1,K], stddev=0.5))
    std_fi = tf.Variable(tf.random_normal(shape=[1,K], stddev=0.5))
    X = tf.placeholder(tf.float32, [None, 100])
    distance = pairwise_distance(X,U)
    log_prior = logsoftmax(FI)
    log_likelihood = -D/2 * tf.log(2.*math.pi*tf.exp(std_fi)) - tf.div(distance,(2.*tf.exp(std_fi)))
    total_loss  = -1 * tf.reduce_sum(reduce_logsumexp(log_prior + log_likelihood, 1, True))
    log_posterior = log_likelihood + log_prior + total_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.99,epsilon=1e-5)
    train = optimizer.minimize(loss=total_loss)
    return train, X, U, FI, std_fi,total_loss, log_posterior

# declear a function to display training process
def runGraph(learning_rate, K, D):
    train_data = data[:6667]
    valid_data = data[6667:]
    train, X, U, FI, std_fi, total_loss,log_posterior = buildGraph(learning_rate, K, D)
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)
    for step in range(0, 3000):
        u, train_process, Fi, Std_Fi, loss, posterior = sess.run([U, train,FI,std_fi, total_loss, log_posterior], feed_dict={X: train_data})
    loss_valid = sess.run(loss_calculation(X,U,std_fi,FI,D),feed_dict={X:valid_data})
    return u, posterior, loss_valid

def cluster_plot(K,D):
    U, log_posterior, loss = runGraph(0.01, K, D)
    index = tf.argmax(log_posterior, 1)
    index = sess.run(index)
    data_list = []
    count = 0
    color = ['r','b','y','g','pink']
    for i in range(0, K):
        for j in range(0, len(index)):
            if index[j] == i:
                data_list.append(data[j])
                count = count + 1
                percentage = count /float(len(index))
        print ('the percentage that data belongs to cluster %s'%i + ' is: %s'%percentage)
        plt.scatter(np.transpose(data_list)[0], np.transpose(data_list)[1], color=color[i])
        data_list = []
        count = 0
        percentage = 0
    plt.scatter(np.transpose(U)[0],np.transpose(U)[1], color = 'k',marker = '*')
    plt.title('scatter plot')
    plt.show()

D = 100
cluster_plot(3,D)

# for k in range(1,50):
#      U, log_posterior, valid_loss = runGraph(0.01,k,D)
#      print ('k = %s,'%k + 'loss of validation set is: %s'%valid_loss)


# def cluster_3():
#     data_list_1 = []
#     data_list_2 = []
#     data_list_3 = []
#     X = tf.placeholder(tf.float32, [None, 2])
#     U, log_posterior = runGraph(0.01, 3)
#     index = tf.argmax(log_posterior, 1)
#     cluster = tf.gather(U,index)
#     index = sess.run(index,feed_dict={X:data})
#     clustered_data = sess.run(cluster, feed_dict={X:data})
#     count = [0,0,0]
#     for i in range(0,len(index)):
#         if index[i] == 0:
#             count[0] = count[0] + 1
#             data_list_1.append(data[i])
#
#         if index[i] == 1:
#             count[1] = count[1] + 1
#             data_list_2.append(data[i])
#
#         if index[i] == 2:
#             count[2] = count[2] + 1
#             data_list_3.append(data[i])
#
#     percentage = count[0]/float(len(index))
#     print ('the percentage that data belongs to cluster 1 is: %s'%percentage)
#     percentage = count[1]/float(len(index))
#     print ('the percentage that data belongs to cluster 2 is: %s'%percentage)
#     percentage = count[2]/float(len(index))
#     print ('the percentage that data belongs to cluster 3 is: %s'%percentage)
#     plt.scatter(np.transpose(data_list_1)[0], np.transpose(data_list_1)[1],color = 'r')
#     plt.scatter(np.transpose(data_list_2)[0], np.transpose(data_list_2)[1], color = 'g')
#     plt.scatter(np.transpose(data_list_3)[0], np.transpose(data_list_3)[1], color = 'y')
#     plt.title('Total Entropy Loss Versus Updating Numbers (minBatchSize = 500)')
#     plt.xlabel('Number of Updates')
#     plt.ylabel('Total loss')
#     plt.show()