import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
sess = tf.Session()
#calculate pairwise distance
def pairwise_distance(X,Z):
    Square_X = tf.reduce_sum(tf.square(X),1)
    Square_Z = tf.reduce_sum(tf.square(Z),1)
    InnerProduct = tf.matmul(X, tf.transpose(Z))
    res = tf.transpose(Square_X + tf.transpose(Square_Z - 2*InnerProduct))
    return res

def log_prob_density(X,U,var,D):
    distance = pairwise_distance(X, U)
    log_gauss = -D / 2 * tf.log(2. * math.pi * tf.square(var)) - tf.div(distance, (2. * tf.square(var)))
    return log_gauss




















#declear a function to display training process
def runGraph(learning_rate,K):
    train, X, U, square_loss, cluster,Index = buildGraph(learning_rate,K)
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)
    for step in range(0,250):
        u,train_process= sess.run([U, train], feed_dict={X:np.load("data2D.npy")})
    u_trained = u
    return u_trained

# k = 1
def cluster_1():
    data = np.load("data2D.npy")
    data_list = []
    X = tf.placeholder(tf.float32, [None, 2])
    U = runGraph(0.001, 1)
    distance = pairwise_distance(X, U)
    index = tf.argmin(distance, 1)
    cluster = tf.gather(U,index)
    index = sess.run(index,feed_dict={X:data})
    clustered_data = sess.run(cluster, feed_dict={X:data})
    count = 0
    for i in range(0,len(index)):
        if index[i] == 0:
            count = count + 1
            data_list.append(data[i])
    percentage = count/float(len(index))
    print ('the percentage is: %s'%percentage)
    plt.scatter(np.transpose(data_list)[0], np.transpose(data_list)[1],label='K = 1')
    plt.title('Total Entropy Loss Versus Updating Numbers (minBatchSize = 500)')
    plt.xlabel('Number of Updates')
    plt.ylabel('Total loss')
    plt.show()

# k = 2
def cluster_2():
    data = np.load("data2D.npy")
    data_list_1 = []
    data_list_2 = []
    X = tf.placeholder(tf.float32, [None, 2])
    U = runGraph(0.001, 2)
    distance = pairwise_distance(X, U)
    index = tf.argmin(distance, 1)
    cluster = tf.gather(U,index)
    index = sess.run(index,feed_dict={X:data})
    clustered_data = sess.run(cluster, feed_dict={X:data})
    count = [0,0]
    for i in range(0,len(index)):
        if index[i] == 0:
            count[0] = count[0] + 1
            data_list_1.append(data[i])

        if index[i] == 1:
            count[1] = count[1] + 1
            data_list_2.append(data[i])

    percentage = count[0]/float(len(index))
    print ('the percentage that data belongs to cluster 1 is: %s'%percentage)
    percentage = count[1]/float(len(index))
    print ('the percentage that data belongs to cluster 2 is: %s'%percentage)
    plt.scatter(np.transpose(data_list_1)[0], np.transpose(data_list_1)[1],color = 'r')
    plt.scatter(np.transpose(data_list_2)[0], np.transpose(data_list_2)[1], color = 'g')
    plt.title('Total Entropy Loss Versus Updating Numbers (minBatchSize = 500)')
    plt.xlabel('Number of Updates')
    plt.ylabel('Total loss')
    plt.show()

#k=3
def cluster_3():
    data = np.load("data2D.npy")
    data_list_1 = []
    data_list_2 = []
    data_list_3 = []
    X = tf.placeholder(tf.float32, [None, 2])
    U = runGraph(0.001, 3)
    distance = pairwise_distance(X, U)
    index = tf.argmin(distance, 1)
    cluster = tf.gather(U,index)
    index = sess.run(index,feed_dict={X:data})
    clustered_data = sess.run(cluster, feed_dict={X:data})
    count = [0,0,0]
    for i in range(0,len(index)):
        if index[i] == 0:
            count[0] = count[0] + 1
            data_list_1.append(data[i])

        if index[i] == 1:
            count[1] = count[1] + 1
            data_list_2.append(data[i])

        if index[i] == 2:
            count[2] = count[2] + 1
            data_list_3.append(data[i])

    percentage = count[0]/float(len(index))
    print ('the percentage that data belongs to cluster 1 is: %s'%percentage)
    percentage = count[1]/float(len(index))
    print ('the percentage that data belongs to cluster 2 is: %s'%percentage)
    percentage = count[2]/float(len(index))
    print ('the percentage that data belongs to cluster 3 is: %s'%percentage)
    plt.scatter(np.transpose(data_list_1)[0], np.transpose(data_list_1)[1],color = 'r')
    plt.scatter(np.transpose(data_list_2)[0], np.transpose(data_list_2)[1], color = 'g')
    plt.scatter(np.transpose(data_list_3)[0], np.transpose(data_list_3)[1], color = 'y')
    plt.title('Total Entropy Loss Versus Updating Numbers (minBatchSize = 500)')
    plt.xlabel('Number of Updates')
    plt.ylabel('Total loss')
    plt.show()


def cluster_4():
    data = np.load("data2D.npy")
    data_list_1 = []
    data_list_2 = []
    data_list_3 = []
    data_list_4 = []
    X = tf.placeholder(tf.float32, [None, 2])
    U = runGraph(0.001, 4)
    distance = pairwise_distance(X, U)
    index = tf.argmin(distance, 1)
    cluster = tf.gather(U,index)
    index = sess.run(index,feed_dict={X:data})
    clustered_data = sess.run(cluster, feed_dict={X:data})
    count = [0,0,0,0]
    for i in range(0,len(index)):
        if index[i] == 0:
            count[0] = count[0] + 1
            data_list_1.append(data[i])

        if index[i] == 1:
            count[1] = count[1] + 1
            data_list_2.append(data[i])

        if index[i] == 2:
            count[2] = count[2] + 1
            data_list_3.append(data[i])
        if index[i] == 3:
            count[3] = count[3] + 1
            data_list_4.append(data[i])

    percentage = count[0]/float(len(index))
    print ('the percentage that data belongs to cluster 1 is: %s'%percentage)
    percentage = count[1]/float(len(index))
    print ('the percentage that data belongs to cluster 2 is: %s'%percentage)
    percentage = count[2]/float(len(index))
    print ('the percentage that data belongs to cluster 3 is: %s'%percentage)
    percentage = count[3]/float(len(index))
    print ('the percentage that data belongs to cluster 4 is: %s'%percentage)
    plt.scatter(np.transpose(data_list_1)[0], np.transpose(data_list_1)[1],color = 'r')
    plt.scatter(np.transpose(data_list_2)[0], np.transpose(data_list_2)[1], color = 'g')
    plt.scatter(np.transpose(data_list_3)[0], np.transpose(data_list_3)[1], color = 'y')
    plt.scatter(np.transpose(data_list_4)[0], np.transpose(data_list_4)[1], color = 'b')
    plt.title('Total Entropy Loss Versus Updating Numbers (minBatchSize = 500)')
    plt.xlabel('Number of Updates')
    plt.ylabel('Total loss')
    plt.show()

#k = 5
def cluster_5():
    data = np.load("data2D.npy")
    data_list_1 = []
    data_list_2 = []
    data_list_3 = []
    data_list_4 = []
    data_list_5 = []
    X = tf.placeholder(tf.float32, [None, 2])
    U = runGraph(0.001, 5)
    distance = pairwise_distance(X, U)
    index = tf.argmin(distance, 1)
    cluster = tf.gather(U,index)
    index = sess.run(index,feed_dict={X:data})


    a = tf.nn.softmax(tf.transpose(cluster))
    b = tf.nn.softmax(tf.transpose(U))
    clustered_data = sess.run(U, feed_dict={X:data})
    #print (sess.run(cluster, feed_dict={X:data}))
    print (sess.run(b, feed_dict={X:data}))
    # count = [0,0,0,0,0]
    # for i in range(0,len(index)):
    #     if index[i] == 0:
    #         count[0] = count[0] + 1
    #         data_list_1.append(data[i])
    #
    #     if index[i] == 1:
    #         count[1] = count[1] + 1
    #         data_list_2.append(data[i])
    #
    #     if index[i] == 2:
    #         count[2] = count[2] + 1
    #         data_list_3.append(data[i])
    #     if index[i] == 3:
    #         count[3] = count[3] + 1
    #         data_list_4.append(data[i])
    #     if index[i] == 4:
    #         count[4] = count[4] + 1
    #         data_list_5.append(data[i])
    #
    # percentage = count[0]/float(len(index))
    # print ('the percentage that data belongs to cluster 1 is: %s'%percentage)
    # percentage = count[1]/float(len(index))
    # print ('the percentage that data belongs to cluster 2 is: %s'%percentage)
    # percentage = count[2]/float(len(index))
    # print ('the percentage that data belongs to cluster 3 is: %s'%percentage)
    # percentage = count[3]/float(len(index))
    # print ('the percentage that data belongs to cluster 4 is: %s'%percentage)
    # percentage = count[4]/float(len(index))
    # print ('the percentage that data belongs to cluster 5 is: %s'%percentage)
    # plt.scatter(np.transpose(data_list_1)[0], np.transpose(data_list_1)[1],color = 'r')
    # plt.scatter(np.transpose(data_list_2)[0], np.transpose(data_list_2)[1], color = 'g')
    # plt.scatter(np.transpose(data_list_3)[0], np.transpose(data_list_3)[1], color = 'y')
    # plt.scatter(np.transpose(data_list_4)[0], np.transpose(data_list_4)[1], color = 'b')
    # plt.scatter(np.transpose(data_list_5)[0], np.transpose(data_list_5)[1], color = 'pink')
    # plt.title('scatter plot')
    # plt.show()


cluster_5()

# a = tf.constant([1,2])
# b = tf.constant([9,5])
# c = a - b
# c = tf.square(c)
# print sess.run(c)
