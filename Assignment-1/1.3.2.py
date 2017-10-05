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
def resp_cal(k,M1,M2,N,ArrayX,ArrayY):
    responsibility = 1/float(k)
    X = tf.placeholder(tf.float64, [M1,N])
    Z = tf.placeholder(tf.float64, [M2,N])
    D = tf.reduce_sum(tf.square(X),1)
    E = tf.reduce_sum(tf.square(Z),1)
    Square_X = tf.reduce_sum(tf.square(X),1)
    Square_Z = tf.reduce_sum(tf.square(Z),1)
    InnerProduct = tf.matmul(X, tf.transpose(Z))
    res = tf.transpose(Square_X + tf.transpose(Square_Z - 2*InnerProduct))
    topK = tf.nn.top_k(-tf.transpose(res), k)
    b1 = topK.indices
    b = sess.run(b1, feed_dict={X:ArrayX, Z:ArrayY})
    index_array = np.linspace(0, M2-1, M2,dtype = int)
    index_array = index_array.repeat(k)
    b = b.reshape(M2*k,)
    result = np.zeros([M2,M1])
    result[index_array,b] = responsibility
    return result

#declare a function to calculate MSE
def MSE_cal(train,trainTarget,input,target,k):
    M1 = np.size(train)
    M2 = np.size(input)
    resp = resp_cal(k,M1,M2,1,train,input)
    error = tf.matmul(tf.transpose(trainTarget),tf.transpose(resp)) - tf.transpose(target)
    MSE = (1/float(2*M2))*tf.reduce_sum(tf.square(error), 1)
    return MSE

#declare a function fro prediction
def predict_function(train, trainTarget,input, k):
    M1 = np.size(train)
    M2 = np.size(input)
    resp = resp_cal(k,M1,M2,1,train,input)
    predictY = tf.matmul(tf.transpose(trainTarget),tf.transpose(resp))
    predict = sess.run(tf.transpose(predictY))
    plt.plot(input, predict,'g')

#display MSE
k = [1,3,5,50]
for i in k:
    print "k=%d, test MSE loss is:"%i
    print sess.run(MSE_cal(trainData,trainTarget,testData,testTarget,i))

X = np.linspace(0.00, 11.0, num = 1000)[:, np.newaxis]

#plot curve
k=50
predict_function(trainData,trainTarget,X,k)
plt.scatter(trainData, trainTarget)
plt.title('k-NN regression on data1D, k=50')
plt.show()

