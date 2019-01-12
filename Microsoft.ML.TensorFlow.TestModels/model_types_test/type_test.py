import tensorflow as tf


f64 = tf.placeholder(dtype=tf.float64, shape=[None,2], name="f64")
f32 = tf.placeholder(dtype=tf.float32, shape=[None,2], name="f32")
i64 = tf.placeholder(dtype=tf.int64, shape=[None,2], name="i64")
i32 = tf.placeholder(dtype=tf.int32, shape=[None,2], name="i32")
i16 = tf.placeholder(dtype=tf.int16, shape=[None,2], name="i16")
i8 = tf.placeholder(dtype=tf.int8, shape=[None,2], name="i8")
u64 = tf.placeholder(dtype=tf.uint64, shape=[None,2], name="u64")
u32 = tf.placeholder(dtype=tf.uint32, shape=[None,2], name="u32")
u16 = tf.placeholder(dtype=tf.uint16, shape=[None,2], name="u16")
u8 = tf.placeholder(dtype=tf.uint8, shape=[None,2], name="u8")
b = tf.placeholder(dtype=tf.bool, shape=[None,2], name="b")

inputs = {'f64':f64, 'f32':f32,
         'i64':i64, 'i32':i32, 'i16':i16, 'i8':i8,
         'u64':u64, 'u32':i32, 'u16':i16, 'u8':i8,
         'b': b}
         

o_f64 = tf.identity(f64, name="o_f64")
o_f32 = tf.identity(f32, name="o_f32")
o_i64 = tf.identity(i64, name="o_i64")
o_i32 = tf.identity(i32, name="o_i32")
o_i16 = tf.identity(i16, name="o_i16")
o_i8 = tf.identity(i8, name="o_i8")
o_u64 = tf.identity(u64, name="o_u64")
o_u32 = tf.identity(u32, name="o_u32")
o_u16 = tf.identity(u16, name="o_u16")
o_u8 = tf.identity(u8, name="o_u8")
o_b = tf.identity(b, name="o_b")

outputs = {'o_f64':o_f64, 'o_f32':o_f32,
         'o_i64':o_i64, 'o_i32':o_i32, 'o_i16':o_i16, 'o_i8':o_i8,
         'o_u64':o_u64, 'o_u32':o_i32, 'o_u16':o_i16, 'o_u8':o_i8,
         'o_b': o_b}
         
# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  tf.saved_model.simple_save(sess, r'model_types_test', inputs=inputs, outputs=outputs )
  