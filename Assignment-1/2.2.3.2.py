import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
sess = tf.Session()

#declear a function to load data
def loadData():
    with np.load("tinymnist.npz") as data:
        trainData, trainTarget = data["x"], data["y"]
        validData, validTarget = data["x_valid"], data["y_valid"]
        testData, testTarget = data["x_test"], data["y_test"]
    return trainData, trainTarget, testData, testTarget, trainData, trainTarget

#declear a function to minimize total loss
def buildGraph(learningRate,decay):
    W = tf.Variable(tf.truncated_normal(shape=[64, 1], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 64], name='input_x')
    y_target = tf.placeholder(tf.float32,[None,1], name='target_y')
    y_predicted = tf.matmul(X, W) + b
    meanSquaredError = tf.reduce_mean(tf.reduce_mean(tf.square(y_predicted - y_target),
                                                     reduction_indices=1,
                                                     name='squared_error'),
                                                     name='mean_squared_error')
    decayLoss =  decay/float(2) * tf.reduce_sum(tf.square(W))
    meanSquaredLoss = meanSquaredError/float(2)
    totalLoss = meanSquaredLoss + decayLoss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    train = optimizer.minimize(loss=totalLoss)
    return W, b, X, y_target, y_predicted, totalLoss, train

#declear a function to display training process
def runMult(batch, learningRate,decay):
    #load data
    W, b, X, y_target, y_predicted, totalLoss, train = buildGraph(learningRate,decay)
    trainData, trainTarget, testData, testTarget, validData, validTarget = loadData()

    #split the dataset according to given batch
    traindata = []
    traintarget = []
    for step in xrange(699):
        if not (step % batch):          #split traindata in 700/batch sets
            traindata.append(validData[step:step + batch, ])
            traintarget.append(validTarget[step:step + batch, ])

    #display training process
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)
    wList = []
    plotloss = []
    predict_acc = []
    miss = 0

    updateNum = 700/batch
    for i in xrange(0,200):
        for step in xrange(0,updateNum):
            _, err, currentW, currentb, yhat= sess.run([train, totalLoss, W, b, y_predicted], feed_dict={X: traindata[step], y_target: traintarget[step]} )
            wList.append(currentW)
            accuracy = 0
            miss = 0
            test_predict = np.dot(testData, currentW) + currentb
            difference = testTarget - test_predict
            for diff in difference:
                if(diff >= 0.5):
                    miss += 1
        accuracy = 1 - miss/float(len(testTarget))
        predict_acc.append(accuracy)
    return predict_acc


plt.subplot(2,1,2)
valid_acc1 = TrainNeuralNetwork(epochs,batch,0.001,decay,1000)[4]
valid_acc2 = TrainNeuralNetwork(epochs,batch,0.001,decay,1000)[4]
valid_acc3 = TrainNeuralNetwork(epochs,batch,0.001,decay,1000)[4]


plt.plot(xArray,valid_acc,label = '')
plt.plot(xArray,valid_acc, label = 'Validation Error')
plt.plot(xArray,test_acc,label = 'Test Error')
plt.title('Total Entropy Loss Versus Number of epochs')
plt.xlabel('Number of epochs')
plt.ylabel('Error')
legend = plt.legend(loc='lower right', shadow=True)
plt.show()



