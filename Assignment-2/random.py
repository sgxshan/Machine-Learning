import tensorflow as tf
import numpy as np
import math
sess = tf.Session()

learning_rate= math.exp(np.random.uniform(-7.5, -4.5))
number_of_layer = int(np.random.uniform(1,5))
number_of_unit = int(np.random.uniform(100,500))
weight_decay = math.exp(np.random.uniform(-9,-6))
wheathe_drop = int(np.random.uniform(0,1))
print learning_rate
print number_of_layer
print number_of_unit
print weight_decay
print wheathe_drop