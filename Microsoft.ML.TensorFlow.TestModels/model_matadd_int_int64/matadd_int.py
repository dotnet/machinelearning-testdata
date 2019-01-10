import tensorflow as tf

# Create some variables.
# v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
# v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

# inc_v1 = v1.assign(v1+1)
# dec_v2 = v2.assign(v2-1)


a = tf.placeholder(dtype=tf.int64, shape=[None,2,2], name="a")# tf.get_variable("a", [None, None], dtype=tf.float32,
  #initializer=tf.zeros_initializer)
  #tf.Variable(1.0, name='a')
b = tf.placeholder(dtype=tf.int32, shape=[None,2,2], name="b") #tf.get_variable("b", [None, None], dtype=tf.float32,
  #initializer=tf.zeros_initializer)
  #tf.Variable(6.0, name='b')

print(a.dtype)
print(b.dtype)


c = tf.add(a, tf.cast(b,dtype=tf.int64), name="c") #tf.matmul(a, b, name="c")
# c = tf.reshape(
    # cc,
    # [-1, 4],
    # name="c"
# )

print(c.dtype)


# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
#saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  
  # Do some work with the model.
  #inc_v1.op.run()
  #dec_v2.op.run()
  
  #sess.run([c], feed_dict=None)
  tf.saved_model.simple_save(sess, r'model_matmul_int', inputs={"a":a, "b":b}, outputs={"c":c} )
  # Save the variables to disk.
  #save_path = saver.save(sess, "model/model.ckpt")
  #print("Model saved in path: %s" % save_path)