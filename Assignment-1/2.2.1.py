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

    #display training process
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)
    wList = []
    plotloss = []
    updateNum = 700/batch
    for i in xrange(0,20):
        # split the dataset according to given batch
        traindata = []
        traintarget = []
        for step in xrange(699):
            if not (step % batch):  # split traindata in 700/batch sets
                traindata.append(validData[step:step + batch, ])
                traintarget.append(validTarget[step:step + batch, ])

        for step in xrange(0,updateNum):
            _, err, currentW, currentb, yhat= sess.run([train, totalLoss, W, b, y_predicted], feed_dict={X: traindata[step], y_target: traintarget[step]} )
            wList.append(currentW)
            plotloss.append(err)
    return plotloss

#2.2.1
plotloss = []
plotNum = []
learning_rate = [0.2,0.1,0.05,0.01]
for rate in learning_rate:
    loss = runMult(50,rate,1)
    plotloss.append(loss)
    numUpdate = len(loss)
    index_array = np.linspace(1, numUpdate, numUpdate, dtype=int)
    plotNum.append(index_array)
plt.plot(plotNum[0],plotloss[0],label = 'learning rate = 0.2')
plt.plot(plotNum[1],plotloss[1],label = 'learning rate = 0.1')
plt.plot(plotNum[2],plotloss[2],label = 'learning rate = 0.05')
plt.plot(plotNum[3],plotloss[3],label = 'learning rate = 0.01')
plt.title('Total Funciton Versus Updating Numbers')
plt.xlabel('Number of Updates')
plt.ylabel('Total loss')
legend = plt.legend(loc='update right', shadow=True)
plt.show()









