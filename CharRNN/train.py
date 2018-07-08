import os
import tensorflow as tf

from read_utils import TextConverter, batch_generator
from model import charRNN

batch_size = 64
seq_len = 50

def main(_):
    model_path = os.path.join('model', 'en')
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    with open("data/shakespeare.txt") as f:
        text = f.read()
    print("=====>", len(text))
    converter = TextConverter(text)
    converter.save(os.path.join(model_path, "converter.pkl"))

    arr = converter.text_to_arr(text)
    g = batch_generator(arr, batch_size, seq_len, converter=None)

    model = charRNN(converter.vocab_size)
    
    model.train(g, model_path)

if __name__ == "__main__":
    # FLAGs variable call "main(args)"
    tf.app.run()