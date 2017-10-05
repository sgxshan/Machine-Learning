import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
sess = tf.Session()

#declear a function to load data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]

        trainData = np.reshape(trainData,[15000, 784])
        testData = np.reshape(testData, [2724, 784])
        validData = np.reshape(validData, [1000, 784])

        traintarget = np.zeros((15000,10))
        index_array1 = np.linspace(0,  14999,15000, dtype=int)
        one_array1 = np.ones(15000,dtype=np.int)
        trainTarget = trainTarget - one_array1

        validtarget = np.zeros((1000,10))
        index_array2 = np.linspace(0,  999,1000, dtype=int)
        one_array2 = np.ones(1000,dtype=np.int)
        validTarget = validTarget - one_array2

        testtarget = np.zeros((2724,10))
        index_array3 = np.linspace(0,  2723,2724, dtype=int)
        one_array3 = np.ones(2724,dtype=np.int)
        testTarget = testTarget - one_array3

        traintarget[index_array1,trainTarget] = 1
        validtarget[index_array2, validTarget] = 1
        testtarget[index_array3, testTarget] = 1
    return trainData, traintarget, testData, testtarget, validData, validtarget

#declear a function to minimize total loss
def buildGraph(learningRate,decay):
    W = tf.Variable(tf.truncated_normal(shape=[784, 10], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32,[None,10], name='target_y')
    y_predicted = tf.matmul(X, W)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(y_predicted + b, y_target)
    decayLoss =  decay/float(2) * tf.reduce_sum(tf.square(W))
    entropyLoss = tf.reduce_mean(cross_entropy)
    totalLoss = entropyLoss + decayLoss
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    train = optimizer.minimize(loss=totalLoss)
    return W, b, X, y_target, y_predicted, totalLoss, train

#declear a function to display training process
def runGraph(batch, learningRate,decay):
    #load data
    W, b, X, y_target, y_predicted, totalLoss, train = buildGraph(learningRate,decay)
    trainData, trainTarget, testData, testTarget, validData, validTarget = loadData()

    #display training process
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    updateNum = 15000/batch
    for i in xrange(0,5):
        # split and randomize the dataset according to given batch
        index = np.arange(0,15000)
        np.random.shuffle(index)
        traindata = []
        traintarget = []
        for step in xrange(14999):
            if not (step % batch):  # split traindata in 700/batch sets
                TrainData = trainData[index]
                TrainTarget = trainTarget[index]
                traindata.append(TrainData[step:step + batch, ])
                traintarget.append(TrainTarget[step:step + batch, ])


        for step in xrange(0,updateNum):
            _, err, yhat= sess.run([train, totalLoss, y_predicted], feed_dict={X: traindata[step], y_target: traintarget[step]} )
            train_loss.append(err)

            # training classification accuracy
            trainpredicted = tf.nn.softmax(yhat)
            hit = tf.equal(tf.argmax(trainpredicted,1), tf.argmax(traintarget[step],1))
            accuracy = tf.reduce_mean(tf.cast(hit,tf.float64)).eval()
            train_acc.append(accuracy)

            err, yhat= sess.run([totalLoss, y_predicted], feed_dict={X: testData, y_target: testTarget} )
            test_loss.append(err)

            # testing classification accuracy
            testpredicted = tf.nn.softmax(yhat)
            hit = tf.equal(tf.argmax(testpredicted,1), tf.argmax(testTarget,1))
            accuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()
            test_acc.append(accuracy)
            print ("Iteration: %d, Test Accuracy: %2f") % (i*step + 1, accuracy)
    return train_loss, test_loss, train_acc, test_acc

plotloss = []
learning_rate = [0.05,0.01,0.001]
for rate in learning_rate:
    train_loss, test_loss, train_acc, test_acc = runGraph(500,rate,0.01)
    plotloss.append(train_loss)

plt.plot(plotloss[0],label = 'learning rate = 0.05')
plt.plot(plotloss[1],label = 'learning rate = 0.01')
plt.plot(plotloss[2],label = 'learning rate = 0.001')
plt.title('Cross Entropy Loss Versus Updating Numbers (minBatchSize = 500)')
plt.xlabel('Number of Updates')
plt.ylabel('Cross Entropy loss')
legend = plt.legend(loc='upper right', shadow=True)
plt.show()

plt.subplot(2,1,1)
train_loss, test_loss,train_acc, test_acc = runGraph(500,0.01,0.01)
plt.plot(train_loss,label = 'Training set')
plt.plot(test_loss,label = 'Testing set')
plt.title('Total Entropy Loss Versus Updating Numbers (minBatchSize = 500)')
plt.ylabel('Total loss')
legend = plt.legend(loc='upper right', shadow=True)

plt.subplot(2,1,2)
plt.plot(train_acc,label = 'Training set')
plt.plot(test_acc,label = 'Testing set')
plt.title('Classification Accuracy Versus Updating Numbers (minBatchSize = 500)')
plt.xlabel('Number of Updates')
plt.ylabel('Classification Accuracy')
legend = plt.legend(loc='lower right', shadow=True)
plt.show()
