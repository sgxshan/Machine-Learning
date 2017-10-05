import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
sess = tf.Session()

#declear a function to load data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx].reshape(-1, 784) / 255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target == posClass] = 1
        Target[Target == negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, trainTarget, testData, testTarget, validData, validTarget

#declear a function to minimize total loss
def buildGraph(learningRate,decay):
    W = tf.Variable(tf.truncated_normal(shape=[784, 1], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float64, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float64,[None,1], name='target_y')
    W = tf.cast(W, tf.float64)
    b = tf.cast(b, tf.float64)
    y_predict = tf.matmul(X, W) + b
    y_predicted = tf.sigmoid(y_predict)

    meanSquaredError = tf.reduce_mean(tf.reduce_mean(tf.square(y_predict - y_target),1),0)
    cross_entropy = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_predict, y_target)))
    decayLoss =  decay/float(2) * tf.reduce_sum(tf.square(W))
    totalLoss = cross_entropy + decayLoss
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    train = optimizer.minimize(loss=totalLoss)
    trainSquaredError = optimizer.minimize(loss = meanSquaredError)
    return W, b, X, y_target, y_predicted, totalLoss,  meanSquaredError, train, trainSquaredError

#declear a function to display training process
def runGraph(batch, learningRate,decay):
    #load data
    W, b, X, y_target, y_predicted, totalLoss, meanSquaredError, train, trainSquaredError = buildGraph(learningRate,decay)
    trainData, trainTarget, testData, testTarget, validData, validTarget = loadData()

    #display training process
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    test_loss = []
    test_acc = []

    train_loss_squared = []
    test_loss_squared = []
    valid_loss_squared = []

    train_acc_squared = []
    test_acc_squared = []
    valid_acc_squared = []

    updateNum = 3500/batch
    for i in xrange(0,100):
        # split and randomize the dataset according to given batch
        index = np.arange(0,3500)
        np.random.shuffle(index)
        traindata = []
        traintarget = []
        for step in xrange(3499):
            if not (step % batch):  # split traindata in 700/batch sets
                TrainData = trainData[index]
                TrainTarget = trainTarget[index]
                traindata.append(TrainData[step:step + batch, ])
                traintarget.append(TrainTarget[step:step + batch, ])


        for step in xrange(0,updateNum):
            _, err, yhat= sess.run([train, totalLoss, y_predicted], feed_dict={X: traindata[step], y_target: traintarget[step]} )
            train_loss.append(err)
            accuracy = 0
            train_miss = 0
            difference = traintarget[step] - yhat
            for diff in difference:
                if (abs(diff) >= 0.5):
                    train_miss += 1
            accuracy = 1 - train_miss / float(len(traintarget[step]))
            train_acc.append(accuracy)


            err, yhat = sess.run([totalLoss, y_predicted], feed_dict={X: validData, y_target: validTarget})
            valid_loss.append(err)
            accuracy = 0
            valid_miss = 0
            difference = validTarget - yhat
            for diff in difference:
                if (abs(diff) >= 0.5):
                    valid_miss += 1
            accuracy = 1 - valid_miss / float(len(validTarget))
            valid_acc.append(accuracy)

            err, yhat = sess.run([totalLoss, y_predicted], feed_dict={X: testData, y_target: testTarget})
            test_loss.append(err)
            accuracy = 0
            test_miss = 0
            difference = testTarget - yhat
            for diff in difference:
                if (abs(diff) >= 0.5):
                    test_miss += 1
            accuracy = 1 - test_miss / float(len(testTarget))
            test_acc.append(accuracy)

            _, err, yhat= sess.run([trainSquaredError, meanSquaredError, y_predicted], feed_dict={X: traindata[step], y_target: traintarget[step]} )
            train_loss_squared.append(err)
            accuracy = 0
            train_miss = 0
            difference = traintarget[step] - yhat
            for diff in difference:
                if (abs(diff) >= 0.5):
                    train_miss += 1
            accuracy = 1 - train_miss / float(len(traintarget[step]))
            train_acc_squared.append(accuracy)

            err, yhat = sess.run([meanSquaredError, y_predicted], feed_dict={X: validData, y_target: validTarget})
            valid_loss_squared.append(err)
            accuracy = 0
            valid_miss = 0
            difference = validTarget - yhat
            for diff in difference:
                if (abs(diff) >= 0.5):
                    valid_miss += 1
            accuracy = 1 - valid_miss / float(len(validTarget))
            valid_acc_squared.append(accuracy)


            err, yhat = sess.run([meanSquaredError, y_predicted], feed_dict={X: testData, y_target: testTarget})
            test_loss_squared.append(err)
            accuracy = 0
            test_miss = 0
            difference = testTarget - yhat
            for diff in difference:
                if (abs(diff) >= 0.5):
                    test_miss += 1
            accuracy = 1 - test_miss / float(len(testTarget))
            test_acc_squared.append(accuracy)

    return train_loss, train_loss_squared, valid_loss, valid_loss_squared, test_loss, test_loss_squared, train_acc, test_acc, train_acc_squared, valid_acc_squared, test_acc_squared, valid_acc

train_loss, train_loss_squared,valid_loss, valid_loss_squared, test_loss, test_loss_squared ,train_acc, test_acc, train_acc_squared, valid_acc_squared, test_acc_squared, valid_acc = runGraph(500,0.01,0)
plt.subplot(3,1,1)
plt.plot(train_acc,label = 'optimal logistic regression (train)')
plt.plot(train_acc_squared,label = 'least squared solution (train)')
plt.title('Classification Accuracy Versus Updating Numbers (minBatchSize = 500)')
plt.ylabel('Classificstion Accuracy')
legend = plt.legend(loc='lower right', shadow=True)

plt.subplot(3,1,2)
plt.plot(valid_acc,label = 'optimal logistic regression (validation)')
plt.plot(valid_acc_squared,label = 'least squared solution (validation)')
plt.ylabel('Accuracy')
legend = plt.legend(loc='lower right', shadow=True)


plt.subplot(3,1,3)
plt.plot(test_acc,label = 'optimal logistic regression (test)')
plt.plot(test_acc_squared,label = 'least squared solution (test)')
plt.xlabel('Number of Updates')
plt.ylabel('Accuracy')
legend = plt.legend(loc='lower right', shadow=True)
plt.show()

#tune learning rate
plotloss = []
learning_rate = [0.1,0.01,0.001]
for rate in learning_rate:
    train_loss, test_loss, train_acc, test_acc = runGraph(500,rate,0.01)[0]
    plotloss.append(train_loss)

plt.plot(plotloss[0],label = 'learning rate = 0.8')
plt.plot(plotloss[1],label = 'learning rate = 0.1')
plt.plot(plotloss[2],label = 'learning rate = 0.01')
plt.title('Total Entropy Loss Versus Updating Numbers (minBatchSize = 500)')
plt.xlabel('Number of Updates')
plt.ylabel('Total loss')
legend = plt.legend(loc='upper right', shadow=True)
plt.show()
