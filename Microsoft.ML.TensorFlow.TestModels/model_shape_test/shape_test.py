import tensorflow as tf


TwoDim = tf.placeholder(dtype=tf.float32, shape=[2, None], name="TwoDim")
ThreeDim = tf.placeholder(dtype=tf.float32, shape=[1, None,2], name="ThreeDim")
FourDim = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name="FourDim")
FourDimKnown = tf.placeholder(dtype=tf.float32, shape=[2, 2, 2, 2], name="FourDimKnown")

inputs = {'TwoDim':TwoDim, 'ThreeDim':ThreeDim,
         'FourDim':FourDim, 'FourDimKnown':FourDimKnown}
         

o_TwoDim = tf.identity(TwoDim, name="o_TwoDim")
o_ThreeDim = tf.identity(ThreeDim, name="o_ThreeDim")
o_FourDim = tf.identity(FourDim, name="o_FourDim")
o_FourDimKnown = tf.identity(FourDimKnown, name="o_FourDimKnown")

outputs = {'o_TwoDim':o_TwoDim, 'o_ThreeDim':o_ThreeDim,
         'o_FourDim':o_FourDim, 'o_FourDimKnown':o_FourDimKnown}
         
# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  tf.saved_model.simple_save(sess, r'model_shape_test', inputs=inputs, outputs=outputs )
  