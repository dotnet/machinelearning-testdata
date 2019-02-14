import tensorflow as tf


A = tf.placeholder(dtype=tf.string, shape=[None, 2], name="A")
B = tf.placeholder(dtype=tf.string, shape=[None], name="B")

inputs = {'A':A, 'B': B}

Original_A = tf.reshape(A, 
    shape=[-1, 2], 
    name="Original_A"
)

splited_B = tf.string_split(
    B,
    delimiter='/')

Joined_Splited_Text = tf.reduce_join(
    splited_B.values,
    separator=' '
)

Joined_Splited_Text = tf.reshape(
    Joined_Splited_Text,
    shape=[-1, 1], 
    name="Joined_Splited_Text"
)

outputs = {'Joined_Splited_Text':Joined_Splited_Text, 'Original_A': Original_A}
         
# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  
  print(sess.run([Original_A, Joined_Splited_Text], feed_dict={A: [["This is fine.", "That's ok."]], B: ["Thank/you/very/much!.", "I/am/grateful/to/you.", "So/nice/of/you."]}))
  tf.saved_model.simple_save(sess, r'model_string_test', inputs=inputs, outputs=outputs )
  