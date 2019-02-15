import tensorflow as tf

a = tf.placeholder(dtype=tf.float32, shape=[None,2,2], name="a")
b = tf.placeholder(dtype=tf.float32, shape=[None,2,2], name="b")

c = tf.matmul(a, b, name="c")

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    tf.saved_model.simple_save(sess, r'model_matmul_int', inputs={"a":a, "b":b}, outputs={"c":c} )