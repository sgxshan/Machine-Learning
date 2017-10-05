import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
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

def NeuralNetwork(input_tensor, num_hid_unit):
    W_hid_layer = tf.Variable(tf.random_normal(shape=(784, num_hid_unit), stddev=3./ (500 + num_hid_unit)),name='W1')
    W_out_layer = tf.Variable(tf.random_normal(shape=(num_hid_unit, 10), stddev=3./ (10 + num_hid_unit)),name='W2')
    b_hid_layer = tf.Variable(0.0, [num_hid_unit,])
    b_out_layer = tf.Variable(0.0, [10,])
    hid_layer_out = tf.nn.relu(tf.matmul(input_tensor, W_hid_layer) + b_hid_layer)
    output = tf.matmul(hid_layer_out,W_out_layer) + b_out_layer
    return W_hid_layer, W_out_layer, output


#declear a function to minimize total loss
def buildGraph(learningRate,decay,num_hid_unit):
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32,[None,10], name='target_y')
    W_hid_layer, W_out_layer, y_predict = NeuralNetwork(X,num_hid_unit)
    y_predicted = tf.sigmoid(y_predict)
    W = tf.reduce_mean(tf.square(W_hid_layer)) + tf.reduce_mean(tf.square(W_out_layer))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_predict, y_target))
    decayLoss =  decay/float(2) * W
    totalLoss = cross_entropy + decayLoss
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    train = optimizer.minimize(loss=totalLoss)
    return W_hid_layer, W_out_layer, X, y_target, y_predicted, totalLoss, train

#declear a function to display training process
def TrainNeuralNetwork(epochs, batch, learningRate,decay,num_hid_unit):
    #load data
    W_hid_layer, W_out_layer, X, y_target, y_predicted, totalLoss, train = buildGraph(learningRate,decay, num_hid_unit)
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
            train_, trainerr, train_yhat= sess.run([train, totalLoss, y_predicted], feed_dict={X: traindata[step2], y_target: traintargets[step2]} )
            validerr, validyhat = sess.run([totalLoss, y_predicted], feed_dict={X: validData, y_target: validtarget} )
            testerr, testyhat = sess.run([totalLoss, y_predicted], feed_dict={X: testData, y_target: testtarget} )

            if (step2 == 0):
                trainpredicted = tf.nn.softmax(train_yhat)
                hit = tf.equal(tf.argmax(trainpredicted, 1), tf.argmax(traintargets[step2], 1))
                trainaccuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()

                validpredicted = tf.nn.softmax(validyhat)
                hit = tf.equal(tf.argmax(validpredicted, 1), tf.argmax(validtarget, 1))
                validaccuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()

                testpredicted = tf.nn.softmax(testyhat)
                hit = tf.equal(tf.argmax(testpredicted, 1), tf.argmax(testtarget, 1))
                testaccuracy = tf.reduce_mean(tf.cast(hit, tf.float64)).eval()

                print ("Epoch: %d, Train Error: %2f") % (i + 1, 1 - trainaccuracy)
                print ("Epoch: %d, Validation Error: %2f") % (i + 1, 1 - validaccuracy)
                print ("Epoch: %d, Test Error: %2f") % (i + 1, 1 - testaccuracy)

                print ("Epoch: %d, Train Loss: %2f") % (i + 1, trainerr)
                print ("Epoch: %d, Validation Loss: %2f") % (i + 1, validerr)
                print ("Epoch: %d, Test Loss: %2f") % (i + 1, testerr)
                trainloss.append(trainerr)
                validloss.append(validerr)
                testloss.append(testerr)

                train_acc.append(1 - trainaccuracy)
                valid_acc.append(1 - validaccuracy)
                test_acc.append(1 - testaccuracy)
    return trainloss, validloss, testloss, train_acc, valid_acc, test_acc

batch =500
epochs = 40
decay = 3*math.exp(-4)
xArray = np.linspace(1, epochs, epochs)

trainloss, validloss, testloss, train_acc, valid_acc1, test_acc = TrainNeuralNetwork(epochs,batch,0.001,decay,100)
trainloss, validloss, testloss, train_acc, valid_acc2, test_acc = TrainNeuralNetwork(epochs,batch,0.001,decay,500)
trainloss, validloss, testloss, train_acc, valid_acc3, test_acc = TrainNeuralNetwork(epochs,batch,0.001,decay,1000)
plt.plot(xArray, valid_acc1,label = 'neural units = 100')
plt.plot(xArray, valid_acc2,label = 'neural units = 500')
plt.plot(xArray, valid_acc3,label = 'neural units = 1000')
plt.title('Classification Error Versus Number of epochs (validation set)')
plt.xlabel('Number of epochs')
plt.ylabel('Classification Error')
legend = plt.legend(loc='upper right', shadow=True)
plt.show()


trainloss, validloss, testloss, train_acc, valid_acc3, test_acc = TrainNeuralNetwork(epochs,batch,0.001,decay,1000)
plt.plot(xArray, test_acc,label = 'Test set')
plt.title('Classification Error Versus Number of epochs (hidden unit = 1000)')
plt.xlabel('Number of epochs')
plt.ylabel('Classification Error')
legend = plt.legend(loc='upper right', shadow=True)
plt.show()









