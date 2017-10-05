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

#declear a function to calculate responsibility matrix
def resp_cal(M1,M2,N,ArrayX,ArrayY):
    X = tf.placeholder(tf.float64, [M1,N])
    Z = tf.placeholder(tf.float64, [M2,N])

    Square_X = tf.reduce_sum(tf.square(X), 1)
    InnerProduct = tf.matmul(X, tf.transpose(X))
    res_XX = tf.transpose(Square_X + tf.transpose(Square_X - 2 * InnerProduct))
    res_XX = tf.exp(-100 * res_XX)

    Square_X = tf.reduce_sum(tf.square(X),1)
    Square_Z = tf.reduce_sum(tf.square(Z),1)
    InnerProduct = tf.matmul(X, tf.transpose(Z))
    res = tf.transpose(Square_X + tf.transpose(Square_Z - 2*InnerProduct))
    res = tf.exp(-100* res)

    res_XX = tf.matrix_inverse(res_XX)
    result=tf.matmul(res_XX, res)
    return sess.run(result, feed_dict={X:ArrayX, Z:ArrayY})

#declear a function to calculate MSE
def MSE_cal(train,trainTarget,input,target):
    M1 = np.size(train)
    M2 = np.size(input)
    resp = resp_cal(M1,M2,1,train,input)
    error = tf.matmul(tf.transpose(trainTarget),resp) - tf.transpose(target)
    MSE = (1/float(2*M2))*tf.reduce_sum(tf.square(error), 1)
    return MSE


#declear a function fro prediction
def predict_function(train, trainTarget,input):
    M1 = np.size(train)
    M2 = np.size(input)
    resp = resp_cal(M1,M2,1,train,input)
    predictY = tf.matmul(tf.transpose(trainTarget),resp)
    predict = sess.run(tf.transpose(predictY))
    print predict
    plt.plot(input, predict,"*")
    plt.plot(trainData, trainTarget,".")
    plt.title('Gaussian process regression')
    plt.show()

#plot curve
predict_function(trainData,trainTarget,testData)



