
# TensorFlow Text Classification 
This model is based on the following TensorFlow model. It was built with version 1.12.0.
https://github.com/tensorflow/models/tree/master/research/sentiment_analysis

The following lines of code needed after the training and evaluation code (but before session is closed) so that model can be saved in [SavedModel](https://www.tensorflow.org/guide/saved_model) format together with word index `imdb_word_index.csv`.

``` python
tf.saved_model.simple_save(
    K.get_session(),
    export_dir = "./saved_model",
    inputs = {"Features": model.inputs[0]},
    outputs = {"Prediction": model.outputs[0]})
  
word_index = tf.keras.datasets.imdb.get_word_index(path='imdb_word_index.json')
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = START_CHAR
word_index["<END>"] = END_CHAR  
word_index["<OOV>"] = OOV_CHAR

with open('./saved_model/imdb_word_index.csv', 'w', encoding="utf-8") as outfile:
  for k,v in word_index.items():
    if v < voc_size + 3:
      outfile.write(k)
      outfile.write(", ")
      outfile.write(str(v))
      outfile.write("\n")
```