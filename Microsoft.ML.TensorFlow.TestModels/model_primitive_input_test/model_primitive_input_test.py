import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

input1 = tf.placeholder(tf.string, name="input1")
input2 = tf.placeholder(tf.string, name="input2")

inputs = {'input1': input1, 'input2': input2}

string_merge = tf.add(input1, input2, name="string_merge")

outputs = {'string_merge':string_merge}

with tf.Session() as sess:
    string_merge = sess.run(string_merge, feed_dict={input1: "This is Text1", input2: "This is Text2"})
    print (string_merge)

    tf.saved_model.simple_save(sess, r'model_primitive_input_test', inputs=inputs, outputs=outputs )
