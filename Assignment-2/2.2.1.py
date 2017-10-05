import tensorflow as tf
def NeuralNetwork(input_tensor, num_hid_unit):
    W_hid_layer = tf.Variable(tf.random_normal(shape=(784, num_hid_unit), stddev=3./ (500 + num_hid_unit)),name='W1')
    W_out_layer = tf.Variable(tf.random_normal(shape=(num_hid_unit, 10), stddev=3./ (10 + num_hid_unit)),name='W2')
    b_hid_layer = tf.Variable(0.0, [num_hid_unit,])
    b_out_layer = tf.Variable(0.0, [10,])
    hid_layer_out = tf.nn.relu(tf.matmul(input_tensor, W_hid_layer) + b_hid_layer)
    output = tf.matmul(hid_layer_out,W_out_layer) + b_out_layer
    return W_hid_layer, W_out_layer, output







