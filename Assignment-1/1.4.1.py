import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
sess = tf.Session()

#load data
np.random.seed(521)
Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) \
         + 0.5 * np.random.randn(100 , 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

#declare a function to calculate responsibility matrix
def resp_cal(M1,M2,N,ArrayX,ArrayY):
    X = tf.placeholder(tf.float64, [M1,N])
    Z = tf.placeholder(tf.float64, [M2,N])

    Square_X = tf.reduce_sum(tf.square(X), 1)
    Square_Z = tf.reduce_sum(tf.square(Z), 1)
    InnerProduct = tf.matmul(X, tf.transpose(Z))
    res_xx = tf.transpose(Square_X + tf.transpose(Square_Z - 2 * InnerProduct))
    res_xx = tf.exp(-100 * res_xx)

    Square_X = tf.reduce_sum(tf.square(X),1)
    Square_Z = tf.reduce_sum(tf.square(Z),1)
    InnerProduct = tf.matmul(X, tf.transpose(Z))
    res = tf.transpose(Square_X + tf.transpose(Square_Z - 2*InnerProduct))
    res = tf.exp(-100* res)

    res_xx = tf.reduce_sum(tf.transpose(res_xx),1)

    print (sess.run(res_xx,feed_dict={X:ArrayX, Z:ArrayY}))
    result=res/res_xx
    return sess.run(result, feed_dict={X:ArrayX, Z:ArrayY})

#declare a function to calculate MSE
def MSE_cal(train,trainTarget,input,target):
    M1 = np.size(train)
    M2 = np.size(input)
    resp = resp_cal(M1,M2,1,train,input)
    error = tf.matmul(tf.transpose(trainTarget),resp) - tf.transpose(target)
    MSE = (1/float(2*M2))*tf.reduce_sum(tf.square(error), 1)
    return MSE


#declare a function fro prediction
def predict_function(train, trainTarget,input):
    M1 = np.size(train)
    M2 = np.size(input)
    resp = resp_cal(M1,M2,1,train,input)
    predictY = tf.matmul(tf.transpose(trainTarget),resp)
    predict = sess.run(tf.transpose(predictY))
    print (predict)
    plt.plot(input, predict,"*")
    plt.plot(trainData, trainTarget,".")
    plt.title('soft k-NN regression')
    plt.show()

X = np.linspace(0.00, 11.0, num = 1000)[:, np.newaxis]

predict_function(trainData,trainTarget,testData)


