import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

input1 = tf.placeholder(tf.string, shape=(None), name="input1")
input2 = tf.placeholder(tf.string, shape=(None), name="input2")

inputs = {'input1': input1, 'input2': input2}

string_same = tf.cond(tf.equal(input1, input2), lambda: tf.constant(True), lambda: tf.constant(False))

outputs = {'string_same':string_same}

with tf.Session() as sess:
    string_same = sess.run(string_same, feed_dict={input1: "This is Text1", input2: "This is Text1"})
    print (string_same)
    
    tf.saved_model.simple_save(sess, r'model_primitive_input_test2', inputs=inputs, outputs=outputs )
