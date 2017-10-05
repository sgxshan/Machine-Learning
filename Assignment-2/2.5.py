import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
sess = tf.Session()

#declear a function to load data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(409)
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


def NeuralNetworkfifth(input_tensor, num_hid_unit):
    W_hid_layer_1 = tf.Variable(tf.random_normal(shape=(784, num_hid_unit), stddev=3./ (500 + num_hid_unit)),name='W1')
    W_hid_layer_2 = tf.Variable(tf.random_normal(shape=(num_hid_unit, num_hid_unit), stddev=3. / (num_hid_unit + num_hid_unit)),name='W2')
    W_hid_layer_3 = tf.Variable(tf.random_normal(shape=(num_hid_unit, num_hid_unit), stddev=3. / (num_hid_unit + num_hid_unit)),name='W3')
    W_hid_layer_4 = tf.Variable(tf.random_normal(shape=(num_hid_unit, num_hid_unit), stddev=3. / (num_hid_unit + num_hid_unit)),name='W4')
    W_hid_layer_5 = tf.Variable(tf.random_normal(shape=(num_hid_unit, num_hid_unit), stddev=3. / (num_hid_unit + num_hid_unit)), name='W5')
    W_out_layer = tf.Variable(tf.random_normal(shape=(num_hid_unit, 10), stddev=3. / (10 + num_hid_unit)),name='W3')

    b_hid_layer_1 = tf.Variable(0.0, [num_hid_unit,])
    b_hid_layer_2 = tf.Variable(0.0, [num_hid_unit,])
    b_hid_layer_3 = tf.Variable(0.0, [num_hid_unit,])
    b_hid_layer_4 = tf.Variable(0.0, [num_hid_unit, ])
    b_hid_layer_5 = tf.Variable(0.0, [num_hid_unit, ])
    b_out_layer = tf.Variable(0.0, [10,])
    hid_layer_out_1 = tf.nn.relu(tf.matmul(input_tensor, W_hid_layer_1) + b_hid_layer_1)
    hid_layer_out_2 = tf.nn.relu(tf.matmul(hid_layer_out_1, W_hid_layer_2) + b_hid_layer_2)
    hid_layer_out_3 = tf.nn.relu(tf.matmul(hid_layer_out_2, W_hid_layer_3) + b_hid_layer_3)
    hid_layer_out_4 = tf.nn.relu(tf.matmul(hid_layer_out_3, W_hid_layer_4) + b_hid_layer_4)
    hid_layer_out_5 = tf.nn.relu(tf.matmul(hid_layer_out_4, W_hid_layer_5) + b_hid_layer_5)
    output = tf.matmul(hid_layer_out_5,W_out_layer) + b_out_layer
    return output, W_hid_layer_1, W_hid_layer_2, W_hid_layer_3, W_hid_layer_4,  W_hid_layer_5, W_out_layer

#declear a function to minimize total loss
def buildGraphfifth(learningRate,decay,num_hid_unit):
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32,[None,10], name='target_y')
    y_predict, W_hid_layer_1, W_hid_layer_2, W_hid_layer_3, W_hid_layer_4,W_hid_layer_5, W_out_layer= NeuralNetworkfifth(X,num_hid_unit)
    W = tf.reduce_mean(tf.square(W_hid_layer_1)) + tf.reduce_mean(tf.square(W_hid_layer_2)) + tf.reduce_mean(tf.square(W_hid_layer_3))+ tf.reduce_mean(tf.square(W_hid_layer_4))+tf.reduce_mean(tf.square(W_out_layer))
    y_predicted = tf.sigmoid(y_predict)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_predict, y_target))
    decayLoss =  decay/float(2) * W
    totalLoss = cross_entropy + decayLoss
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    train = optimizer.minimize(loss=totalLoss)
    return X, y_target, y_predicted, totalLoss, train

def TrainNeuralNetworkfifth(epochs,batch, learningRate,decay,num_hid_unit):
    #load data
    X, y_target, y_predicted, totalLoss, train = buildGraphfifth(learningRate,decay, num_hid_unit)
    trainData, traintarget, testData, testtarget, validData, validtarget = loadData()

    #display training process
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)
    trainloss = []
    validloss = []
    testloss = []
    train_acc = []
    valid_acc = []
    test_acc = []
    updateNUm = 15000/batch
    for i in xrange(0,epochs):
        # split the dataset according to given batch
        traindata = []
        traintargets = []

        for step in xrange(14999):
            if not (step % batch):  # split traindata in 15000/batch sets
                traindata.append(trainData[step:step + batch, ])
                traintargets.append(traintarget[step:step + batch, ])

        for step2 in xrange(0,updateNUm):
            train_, trainerr, trainyhat= sess.run([train, totalLoss, y_predicted], feed_dict={X: traindata[step2], y_target: traintargets[step2]} )
            validerr, validyhat = sess.run([totalLoss, y_predicted], feed_dict={X: validData, y_target: validtarget} )
            testerr, testyhat = sess.run([totalLoss, y_predicted], feed_dict={X: testData, y_target: testtarget} )
            if(step2==0):
                trainpredicted = tf.nn.softmax(trainyhat)
                hit = tf.equal(tf.argmax(trainpredicted, 1), tf.argmax(traintargets[step2], 1))
                trainaccuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()

                validpredicted = tf.nn.softmax(validyhat)
                hit = tf.equal(tf.argmax(validpredicted, 1), tf.argmax(validtarget, 1))
                validaccuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()

                testpredicted = tf.nn.softmax(testyhat)
                hit = tf.equal(tf.argmax(testpredicted, 1), tf.argmax(testtarget, 1))
                testaccuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()

                print ("Epochs: %d, Validation Error: %2f")%(i+1, 1-validaccuracy)
                trainloss.append(trainerr)
                validloss.append(validerr)
                testloss.append(testerr)
                train_acc.append(1-trainaccuracy)
                valid_acc.append(1-validaccuracy)
                test_acc.append(1-testaccuracy)
    return trainloss, validloss, testloss, train_acc, valid_acc, test_acc


def NeuralNetworkfourth(input_tensor, num_hid_unit):
    W_hid_layer_1 = tf.Variable(tf.random_normal(shape=(784, num_hid_unit), stddev=3./ (500 + num_hid_unit)),name='W1')
    W_hid_layer_2 = tf.Variable(tf.random_normal(shape=(num_hid_unit, num_hid_unit), stddev=3. / (num_hid_unit + num_hid_unit)),name='W2')
    W_hid_layer_3 = tf.Variable(tf.random_normal(shape=(num_hid_unit, num_hid_unit), stddev=3. / (num_hid_unit + num_hid_unit)),name='W3')
    W_hid_layer_4 = tf.Variable(tf.random_normal(shape=(num_hid_unit, num_hid_unit), stddev=3. / (num_hid_unit + num_hid_unit)),name='W4')
    W_out_layer = tf.Variable(tf.random_normal(shape=(num_hid_unit, 10), stddev=3. / (10 + num_hid_unit)),name='W3')

    b_hid_layer_1 = tf.Variable(0.0, [num_hid_unit,])
    b_hid_layer_2 = tf.Variable(0.0, [num_hid_unit,])
    b_hid_layer_3 = tf.Variable(0.0, [num_hid_unit,])
    b_hid_layer_4 = tf.Variable(0.0, [num_hid_unit, ])
    b_out_layer = tf.Variable(0.0, [10,])
    hid_layer_out_1 = tf.nn.relu(tf.matmul(input_tensor, W_hid_layer_1) + b_hid_layer_1)
    hid_layer_out_2 = tf.nn.relu(tf.matmul(hid_layer_out_1, W_hid_layer_2) + b_hid_layer_2)
    hid_layer_out_3 = tf.nn.relu(tf.matmul(hid_layer_out_2, W_hid_layer_3) + b_hid_layer_3)
    hid_layer_out_4 = tf.nn.relu(tf.matmul(hid_layer_out_3, W_hid_layer_4) + b_hid_layer_4)
    output = tf.matmul(hid_layer_out_4,W_out_layer) + b_out_layer
    return output, W_hid_layer_1, W_hid_layer_2, W_hid_layer_3, W_hid_layer_4, W_out_layer

#declear a function to minimize total loss
def buildGraphfourth(learningRate,decay,num_hid_unit):
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32,[None,10], name='target_y')
    y_predict, W_hid_layer_1, W_hid_layer_2, W_hid_layer_3, W_hid_layer_4, W_out_layer= NeuralNetworkfourth(X,num_hid_unit)
    W = tf.reduce_mean(tf.square(W_hid_layer_1)) + tf.reduce_mean(tf.square(W_hid_layer_2)) + tf.reduce_mean(tf.square(W_hid_layer_3))+ tf.reduce_mean(tf.square(W_hid_layer_4))+tf.reduce_mean(tf.square(W_out_layer))
    y_predicted = tf.sigmoid(y_predict)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_predict, y_target))
    decayLoss =  decay/float(2) * W
    totalLoss = cross_entropy + decayLoss
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    train = optimizer.minimize(loss=totalLoss)
    return X, y_target, y_predicted, totalLoss, train

def TrainNeuralNetworkfourth(epochs,batch, learningRate,decay,num_hid_unit):
    #load data
    X, y_target, y_predicted, totalLoss, train = buildGraphfourth(learningRate,decay, num_hid_unit)
    trainData, traintarget, testData, testtarget, validData, validtarget = loadData()

    #display training process
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)
    trainloss = []
    validloss = []
    testloss = []
    train_acc = []
    valid_acc = []
    test_acc = []
    updateNUm = 15000/batch
    for i in xrange(0,epochs):
        # split the dataset according to given batch
        traindata = []
        traintargets = []

        for step in xrange(14999):
            if not (step % batch):  # split traindata in 15000/batch sets
                traindata.append(trainData[step:step + batch, ])
                traintargets.append(traintarget[step:step + batch, ])

        for step2 in xrange(0,updateNUm):
            train_, trainerr, trainyhat= sess.run([train, totalLoss, y_predicted], feed_dict={X: traindata[step2], y_target: traintargets[step2]} )
            validerr, validyhat = sess.run([totalLoss, y_predicted], feed_dict={X: validData, y_target: validtarget} )
            testerr, testyhat = sess.run([totalLoss, y_predicted], feed_dict={X: testData, y_target: testtarget} )
            if(step2==0):
                trainpredicted = tf.nn.softmax(trainyhat)
                hit = tf.equal(tf.argmax(trainpredicted, 1), tf.argmax(traintargets[step2], 1))
                trainaccuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()

                validpredicted = tf.nn.softmax(validyhat)
                hit = tf.equal(tf.argmax(validpredicted, 1), tf.argmax(validtarget, 1))
                validaccuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()

                testpredicted = tf.nn.softmax(testyhat)
                hit = tf.equal(tf.argmax(testpredicted, 1), tf.argmax(testtarget, 1))
                testaccuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()
                print ("Epochs: %d, Test Error: %2f") % (i + 1, 1 - testaccuracy)
                print ("Epochs: %d, Validation Error: %2f")%(i+1, 1-validaccuracy)
                trainloss.append(trainerr)
                validloss.append(validerr)
                testloss.append(testerr)
                train_acc.append(1-trainaccuracy)
                valid_acc.append(1-validaccuracy)
                test_acc.append(1-testaccuracy)
    return trainloss, validloss, testloss, train_acc, valid_acc, test_acc

def NeuralNetworktri(input_tensor, num_hid_unit):
    W_hid_layer_1 = tf.Variable(tf.random_normal(shape=(784, num_hid_unit), stddev=3./ (500 + num_hid_unit)),name='W1')
    W_hid_layer_2 = tf.Variable(tf.random_normal(shape=(num_hid_unit, num_hid_unit), stddev=3. / (500 + num_hid_unit)),name='W2')
    W_hid_layer_3 = tf.Variable(tf.random_normal(shape=(num_hid_unit, num_hid_unit), stddev=3. / (500 + num_hid_unit)),name='W3')
    W_out_layer = tf.Variable(tf.random_normal(shape=(num_hid_unit, 10), stddev=3. / (10 + num_hid_unit)),name='W3')

    b_hid_layer_1 = tf.Variable(0.0, [num_hid_unit,])
    b_hid_layer_2 = tf.Variable(0.0, [num_hid_unit,])
    b_hid_layer_3 = tf.Variable(0.0, [num_hid_unit,])
    b_out_layer = tf.Variable(0.0, [10,])
    hid_layer_out_1 = tf.nn.relu(tf.matmul(input_tensor, W_hid_layer_1) + b_hid_layer_1)
    hid_layer_out_2 = tf.nn.relu(tf.matmul(hid_layer_out_1, W_hid_layer_2) + b_hid_layer_2)
    hid_layer_out_3 = tf.nn.relu(tf.matmul(hid_layer_out_2, W_hid_layer_3) + b_hid_layer_3)
    output = tf.matmul(hid_layer_out_3,W_out_layer) + b_out_layer
    return output, W_hid_layer_1, W_hid_layer_2, W_hid_layer_3,W_out_layer

#declear a function to minimize total loss
def buildGraphtri(learningRate,decay,num_hid_unit):
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32,[None,10], name='target_y')
    y_predict, W_hid_layer_1, W_hid_layer_2, W_hid_layer_3, W_out_layer= NeuralNetworktri(X,num_hid_unit)
    W = tf.reduce_mean(tf.square(W_hid_layer_1)) + tf.reduce_mean(tf.square(W_hid_layer_2)) + tf.reduce_mean(tf.square(W_hid_layer_3))+tf.reduce_mean(tf.square(W_out_layer))
    y_predicted = tf.sigmoid(y_predict)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_predict, y_target))
    decayLoss =  decay/float(2) * W
    totalLoss = cross_entropy + decayLoss
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    train = optimizer.minimize(loss=totalLoss)
    return X, y_target, y_predicted, totalLoss, train

def TrainNeuralNetworktri(epochs,batch, learningRate,decay,num_hid_unit):
    #load data
    X, y_target, y_predicted, totalLoss, train = buildGraphtri(learningRate,decay, num_hid_unit)
    trainData, traintarget, testData, testtarget, validData, validtarget = loadData()

    #display training process
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)
    trainloss = []
    validloss = []
    testloss = []
    train_acc = []
    valid_acc = []
    test_acc = []
    updateNUm = 15000/batch
    for i in xrange(0,epochs):
        # split the dataset according to given batch
        traindata = []
        traintargets = []

        for step in xrange(14999):
            if not (step % batch):  # split traindata in 15000/batch sets
                traindata.append(trainData[step:step + batch, ])
                traintargets.append(traintarget[step:step + batch, ])

        for step2 in xrange(0,updateNUm):
            train_, trainerr, trainyhat= sess.run([train, totalLoss, y_predicted], feed_dict={X: traindata[step2], y_target: traintargets[step2]} )
            validerr, validyhat = sess.run([totalLoss, y_predicted], feed_dict={X: validData, y_target: validtarget} )
            testerr, testyhat = sess.run([totalLoss, y_predicted], feed_dict={X: testData, y_target: testtarget} )
            if(step2==0):
                trainpredicted = tf.nn.softmax(trainyhat)
                hit = tf.equal(tf.argmax(trainpredicted, 1), tf.argmax(traintargets[step2], 1))
                trainaccuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()

                validpredicted = tf.nn.softmax(validyhat)
                hit = tf.equal(tf.argmax(validpredicted, 1), tf.argmax(validtarget, 1))
                validaccuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()

                testpredicted = tf.nn.softmax(testyhat)
                hit = tf.equal(tf.argmax(testpredicted, 1), tf.argmax(testtarget, 1))
                testaccuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()
                print ("Epochs: %d, Test Error: %2f") % (i + 1, 1 - testaccuracy)
                print ("Epochs: %d, Validation Error: %2f")%(i+1, 1-validaccuracy)
                trainloss.append(trainerr)
                validloss.append(validerr)
                testloss.append(testerr)
                train_acc.append(1-trainaccuracy)
                valid_acc.append(1-validaccuracy)
                test_acc.append(1-testaccuracy)
    return trainloss, validloss, testloss, train_acc, valid_acc, test_acc



def NeuralNetwork(input_tensor, num_hid_unit):
    W_hid_layer_1 = tf.Variable(tf.random_normal(shape=(784, num_hid_unit), stddev=3./ (500 + num_hid_unit)),name='W1')
    W_hid_layer_2 = tf.Variable(tf.random_normal(shape=(num_hid_unit, num_hid_unit), stddev=3. / (500 + num_hid_unit)),name='W2')
    W_out_layer = tf.Variable(tf.random_normal(shape=(num_hid_unit, 10), stddev=3. / (10 + num_hid_unit)),name='W3')

    b_hid_layer_1 = tf.Variable(0.0, [num_hid_unit,])
    b_hid_layer_2 = tf.Variable(0.0, [num_hid_unit,])
    b_out_layer = tf.Variable(0.0, [10,])
    hid_layer_out_1 = tf.nn.relu(tf.matmul(input_tensor, W_hid_layer_1) + b_hid_layer_1)
    hid_layer_out_2 = tf.nn.relu(tf.matmul(hid_layer_out_1, W_hid_layer_2) + b_hid_layer_2)
    output = tf.matmul(hid_layer_out_2,W_out_layer) + b_out_layer
    return output, W_hid_layer_1, W_hid_layer_2, W_out_layer

#declear a function to minimize total loss
def buildGraph(learningRate,decay,num_hid_unit):
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32,[None,10], name='target_y')
    y_predict, W_hid_layer_1, W_hid_layer_2, W_out_layer= NeuralNetwork(X,num_hid_unit)
    W = tf.reduce_mean(tf.square(W_hid_layer_1)) + tf.reduce_mean(tf.square(W_hid_layer_2)) + tf.reduce_mean(tf.square(W_out_layer))
    y_predicted = tf.sigmoid(y_predict)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_predict, y_target))
    decayLoss =  decay/float(2) * W
    totalLoss = cross_entropy + decayLoss
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    train = optimizer.minimize(loss=totalLoss)
    return X, y_target, y_predicted, totalLoss, train

def TrainNeuralNetwork(epochs,batch, learningRate,decay,num_hid_unit):
    #load data
    X, y_target, y_predicted, totalLoss, train = buildGraph(learningRate,decay, num_hid_unit)
    trainData, traintarget, testData, testtarget, validData, validtarget = loadData()

    #display training process
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)
    trainloss = []
    validloss = []
    testloss = []
    train_acc = []
    valid_acc = []
    test_acc = []
    updateNUm = 15000/batch
    for i in xrange(0,epochs):
        # split the dataset according to given batch
        traindata = []
        traintargets = []

        for step in xrange(14999):
            if not (step % batch):  # split traindata in 15000/batch sets
                traindata.append(trainData[step:step + batch, ])
                traintargets.append(traintarget[step:step + batch, ])

        for step2 in xrange(0,updateNUm):
            train_, trainerr, trainyhat= sess.run([train, totalLoss, y_predicted], feed_dict={X: traindata[step2], y_target: traintargets[step2]} )
            validerr, validyhat = sess.run([totalLoss, y_predicted], feed_dict={X: validData, y_target: validtarget} )
            testerr, testyhat = sess.run([totalLoss, y_predicted], feed_dict={X: testData, y_target: testtarget} )
            if(step2==0):
                trainpredicted = tf.nn.softmax(trainyhat)
                hit = tf.equal(tf.argmax(trainpredicted, 1), tf.argmax(traintargets[step2], 1))
                trainaccuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()

                validpredicted = tf.nn.softmax(validyhat)
                hit = tf.equal(tf.argmax(validpredicted, 1), tf.argmax(validtarget, 1))
                validaccuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()

                testpredicted = tf.nn.softmax(testyhat)
                hit = tf.equal(tf.argmax(testpredicted, 1), tf.argmax(testtarget, 1))
                testaccuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()

                print ("Epochs: %d, Validation Error: %2f")%(i+1, 1-validaccuracy)
                trainloss.append(trainerr)
                validloss.append(validerr)
                testloss.append(testerr)
                train_acc.append(1-trainaccuracy)
                valid_acc.append(1-validaccuracy)
                test_acc.append(1-testaccuracy)
    return trainloss, validloss, testloss, train_acc, valid_acc, test_acc


def NeuralNetworksingle(input_tensor, num_hid_unit):
    W_hid_layer = tf.Variable(tf.random_normal(shape=(784, num_hid_unit), stddev=3. / (500 + num_hid_unit)),name='W1')
    W_out_layer = tf.Variable(tf.random_normal(shape=(num_hid_unit, 10), stddev=3. / (10 + num_hid_unit)),name='W2')
    b_hid_layer = tf.Variable(0.0, [num_hid_unit,])
    b_out_layer = tf.Variable(0.0, [10,])
    hid_layer_out = tf.nn.relu(tf.matmul(input_tensor, W_hid_layer) + b_hid_layer)
    output = tf.matmul(hid_layer_out,W_out_layer) + b_out_layer
    return W_hid_layer, W_out_layer, output

#declear a function to minimize total loss
def buildGraphsingle(learningRate,decay,num_hid_unit):
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32,[None,10], name='target_y')
    W_hid_layer, W_out_layer, y_predict = NeuralNetworksingle(X,num_hid_unit)
    W = tf.reduce_mean(tf.square(W_hid_layer)) + tf.reduce_mean(tf.square(W_out_layer))
    y_predicted = tf.sigmoid(y_predict)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_predict, y_target))
    decayLoss =  decay/float(2) * W
    totalLoss = cross_entropy + decayLoss
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    train = optimizer.minimize(loss=totalLoss)
    return W_hid_layer, W_out_layer, X, y_target, y_predicted, totalLoss, train

#declear a function to display training process

def TrainNeuralNetworksingle(epochs, batch, learningRate,decay,num_hid_unit):
    #load data
    W_hid_layer, W_out_layer, X, y_target, y_predicted, totalLoss, train = buildGraphsingle(learningRate,decay, num_hid_unit)
    trainData, traintarget, testData, testtarget, validData, validtarget = loadData()

    #display training process
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)
    trainloss = []
    validloss = []
    testloss = []
    train_acc = []
    valid_acc = []
    test_acc = []
    updateNUm = 15000/batch
    for i in xrange(0,epochs):
        # split the dataset according to given batch
        traindata = []
        traintargets = []

        for step in xrange(14999):
            if not (step % batch):  # split traindata in 15000/batch sets
                traindata.append(trainData[step:step + batch, ])
                traintargets.append(traintarget[step:step + batch, ])

        for step2 in xrange(0,updateNUm):
            train_, trainerr, trainyhat= sess.run([train, totalLoss, y_predicted], feed_dict={X: traindata[step2], y_target: traintargets[step2]} )
            validerr, validyhat = sess.run([totalLoss, y_predicted], feed_dict={X: validData, y_target: validtarget} )
            testerr, testyhat = sess.run([totalLoss, y_predicted], feed_dict={X: testData, y_target: testtarget} )
            if step2 ==1:
                trainpredicted = tf.nn.softmax(trainyhat)
                hit = tf.equal(tf.argmax(trainpredicted, 1), tf.argmax(traintargets[step2], 1))
                trainaccuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()

                validpredicted = tf.nn.softmax(validyhat)
                hit = tf.equal(tf.argmax(validpredicted, 1), tf.argmax(validtarget, 1))
                validaccuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()

                testpredicted = tf.nn.softmax(testyhat)
                hit = tf.equal(tf.argmax(testpredicted, 1), tf.argmax(testtarget, 1))
                testaccuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()

                print ("Epochs: %d, Test Error: %2f")%(i+1, 1-testaccuracy)
                print ("Epochs: %d, validation Error: %2f") % (i + 1, 1 - validaccuracy)
                trainloss.append(trainerr)
                validloss.append(validerr)
                testloss.append(testerr)
                train_acc.append(1-trainaccuracy)
                valid_acc.append(1-validaccuracy)
                test_acc.append(1-testaccuracy)
    return trainloss, validloss, testloss, train_acc, valid_acc, test_acc


batch =500
epochs = 10
decay = 3*math.exp(-4)
xArray = np.linspace(1, epochs, epochs)


trainloss, validloss, testloss, train_acc, valid_acc, test_acc = TrainNeuralNetworkfourth(epochs,500,0.0090707,0.000188,289)

