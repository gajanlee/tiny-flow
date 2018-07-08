import tensorflow as tf
from read_utils import TextConverter
from model import charRNN
import os

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer("hidden_size", 128, "size of hidden state of rnn")
tf.flags.DEFINE_integer("num_layers", 2, "number of rnn layers")
tf.flags.DEFINE_string("converter_path", "./model/en/converter.pkl", "modle vocabulary path")
tf.flags.DEFINE_string("checkpoint_path", "./model/en", "checkpoint path")
tf.flags.DEFINE_string("start_string", "I am your", "start string to generate")
tf.flags.DEFINE_integer("max_length", 300, "max length to generate")

def main(_):
    converter = TextConverter(filename=FLAGS.converter_path)
    
    model = charRNN(converter.vocab_size, train=False)
    model.load(tf.train.latest_checkpoint(FLAGS.checkpoint_path))

    start = converter.text_to_arr(FLAGS.start_string)
    arr = model.generate(FLAGS.max_length, start, converter.vocab_size)
    print(converter.arr_to_text(arr))

if __name__ == "__main__":
    tf.app.run()