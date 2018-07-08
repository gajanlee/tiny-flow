import pickle
import numpy as np
from copy import copy

def batch_generator(arr, batch_size, seq_len, converter=None):
    
    arr = copy(arr)
    per_batch = batch_size * seq_len
    n_batches = len(arr) // per_batch
    arr = arr[:per_batch * n_batches]
    arr = arr.reshape((batch_size, -1))
    
    print(n_batches, per_batch, arr.shape)

    while True:
        np.random.shuffle(arr)
        for n in range(0, arr.shape[1]):
            x = arr[:, n:n+seq_len]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            #print(x.shape, y.shape)
            if converter:
                print([converter.int_to_word(_x) for _x in x[0]])
                print([converter.int_to_word(_x) for _x in y[0]])
                return
            if x.shape != (batch_size, seq_len):
                continue
            yield x, y
        print("one passage")


class TextConverter(object):
    def __init__(self, text=None, filename=None, max_vocab=5000):
        if filename:
            with open(filename, "rb") as f:
                self.vocab = pickle.load(f)
        else:
            
            # vocab_count = {word: 0 for word in set(text)}
            # print("[!] Vocabulary size is %s" % len(vocab_count))
            self.vocab = list(set(text))
            print("[!] Vocabulary size is %s" % len(self.vocab))
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        # contains "<unknown>" tag
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index > len(self.vocab) or index < 0:
            raise Exception("Unknown index")
        return self.int_to_word_table[index] if index != len(self.vocab) else "<unk>"
 
    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)
    
    def arr_to_text(self, arr):
        return "".join([self.int_to_word(index) for index in arr])

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.vocab, f)