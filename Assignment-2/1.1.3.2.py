import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
sess = tf.Session()

y_predicted = np.linspace(0, 1, 100)
crossEntropy = -np.log(1-y_predicted)
squaredLoss = np.square(y_predicted)

plt.plot(y_predicted, crossEntropy, label = 'cross entropy loss function')
plt.plot(y_predicted, squaredLoss, label = 'squared loss function')
plt.title('Comparision of two loss functions')
plt.xlabel('predicted value')
plt.ylabel('Total loss')
legend = plt.legend(loc='upper right', shadow=True)
plt.show()
