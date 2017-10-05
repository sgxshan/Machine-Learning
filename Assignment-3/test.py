import tensorflow as tf
sess = tf.Session()
a = tf.constant([[2.,0.,0.],[0.,2.,2.],[0.,0.,2.]])
log_det = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(a))))
print sess.run(a)
print sess.run(log_det)





