import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
sess = tf.Session()
np.random.seed(521)
data = np.load("x.npy")
label = np.load("y.npy")

# print (data)
# print (np.shape(data))
# print (label)
# print (np.shape(label))
# # print (Target)
# # print (np.shape(Target))
# data = np.reshape(data,[700,8,8])
# for i in xrange(1, 100):
#     fig = plt.subplot(10, 10, i)
#     plt.imshow(data[i - 1], cmap=plt.cm.gray)
#     plt.axis('off')
# plt.show()

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

# def log_prob_density(X,U,std_fi,pi,D):
#     distance = pairwise_distance(X,U)
#     log_likelihood = -D/2 * tf.log(2.*math.pi*tf.exp(std_fi)) - tf.div(distance,(2.*tf.exp(std_fi)))
#     log_prior = logsoftmax(pi)
#     log_magrinal = tf.reduce_sum(reduce_logsumexp(log_prior + log_likelihood, 1, True))
#     log_posterior = log_likelihood + log_prior - log_magrinal
#     return log_posterior
#
# def loss_calculation(X,U,std_fi,FI,D):
#     distance = pairwise_distance(X,U)
#     log_prior = logsoftmax(FI)
#     log_likelihood = -D/2 * tf.log(2.*math.pi*tf.exp(std_fi)) - tf.div(distance,(2.*tf.exp(std_fi)))
#     total_loss  = -1 * tf.reduce_sum(reduce_logsumexp(log_prior + log_likelihood, 1, True))
#     return total_loss

#declear a function to minimize total loss
def buildGraph(learning_rate,K, D):
    U = tf.Variable(tf.random_normal(shape=[K, D], stddev=0.5))
    FI = tf.Variable(tf.random_normal(shape=[D,D], stddev=0.5))
    W = tf.Variable(tf.random_normal(shape=[D,K], stddev=0.5))
    X = tf.placeholder(tf.float32, [None, D])
    distance = pairwise_distance(X,U)
    covariance = FI + tf.matmul(W, tf.transpose(W))
    log_det = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(covariance))))
    total_loss = tf.reduce_sum(-D/2 * tf.log(2.*math.pi*log_det) - tf.div(distance,(2.*log_det)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.99,epsilon=1e-5)
    train = optimizer.minimize(loss=total_loss)
    return train, X, U, FI, W,total_loss,

# declear a function to display training process
def runGraph(learning_rate, K, D):
    train, X, U, FI, w, total_loss= buildGraph(learning_rate, K, D)
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)
    loss_list = []
    for step in range(0, 3000):
        u, train_process, Fi, weight, loss= sess.run([U, train,FI, w, total_loss], feed_dict={X: data})
        loss_list.append(loss)
    return loss_list

learning_rate = 0.01
K = 4
D = 64
total_loss = runGraph(learning_rate,K,D)

print (total_loss)
plt.plot(total_loss,label = 'learning rate = 0.001')
plt.show()
plt.title('Total Entropy Loss Versus Updating Numbers (minBatchSize = 500)')
plt.xlabel('Number of Updates')
plt.ylabel('Total loss')

# def cluster_plot(K,D):
#     U, log_posterior, loss = runGraph(0.01, K, D)
#     index = tf.argmax(log_posterior, 1)
#     index = sess.run(index)
#     data_list = []
#     count = 0
#     color = ['r','b','y','g','pink']
#     for i in range(0, K):
#         for j in range(0, len(index)):
#             if index[j] == i:
#                 data_list.append(data[j])
#                 count = count + 1
#                 percentage = count /float(len(index))
#         print ('the percentage that data belongs to cluster %s'%i + ' is: %s'%percentage)
#         plt.scatter(np.transpose(data_list)[0], np.transpose(data_list)[1], color=color[i])
#         data_list = []
#         count = 0
#         percentage = 0
#     plt.scatter(np.transpose(U)[0],np.transpose(U)[1], color = 'k',marker = '*')
#     plt.title('scatter plot')
#     plt.show()

#cluster_plot(3,D)

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