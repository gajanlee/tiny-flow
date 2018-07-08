import tensorflow as tf
import numpy as np
import time, os

def pick_top_n(preds, num_classes, top_n=5):
    p = np.squeeze(preds)

    c = np.random.choice(num_classes, 1, p=p)[0]
    return c

class charRNN:
    def __init__(self, num_classes, batch_size=64, seq_len=50, train=True):
        
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = 128
        self.layers = 2
        self.learning_rate = 0.001
        self.grad_clip = 5

        if not train:
            self.batch_size = self.seq_len = 1

        self.build_inputs()
        self.build_rnn()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_inputs(self):
        with tf.name_scope("inputs"):
            self.inputs = tf.placeholder(tf.int32, shape=(
                self.batch_size, self.seq_len), name="inputs")
            self.targets = tf.placeholder(tf.int32, shape=(
                self.batch_size, self.seq_len), name="targets")
            # self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            self.keep_prob = 1.
            self.rnn_inputs = tf.one_hot(self.inputs, self.num_classes)
            print(self.rnn_inputs)

    def build_rnn(self):

        def get_rnn_cell(hidden_size, keep_prob):
            cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

        with tf.name_scope("rnn"):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_rnn_cell(self.hidden_size, self.keep_prob) for _ in range(self.layers)])
            self.initial_state = cell.zero_state(self.batch_size, tf.float32)
            self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.rnn_inputs, initial_state=self.initial_state)

            seq_output = tf.concat(self.rnn_outputs, 1)
            x = tf.reshape(seq_output, [-1, self.hidden_size])

            with tf.variable_scope("softmax"):
                softmax_w = tf.Variable(tf.truncated_normal([self.hidden_size, self.num_classes]))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))
            
            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.proba_prediction = tf.nn.softmax(self.logits)
    
    def build_loss(self):
        with tf.name_scope("loss"):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)
    
    def build_optimizer(self):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    def train(self, batch_generator, save_path, log_step=10, save_step=1000, max_steps=100000):
        # load model
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            new_state = sess.run(self.initial_state)

            for x, y in batch_generator:
                step += 1
                start = time.time()
                feed = {
                    self.inputs: x,
                    self.targets: y,
                    self.initial_state: new_state
                }
                batch_loss, new_state, _ = sess.run([self.loss, self.final_state, self.optimizer], feed_dict=feed)
                end = time.time()

                if step % log_step == 0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if step % save_step == 0:
                    self.saver.save(sess, os.path.join(save_path, "model"), global_step=step)
                
                if step >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, "model"), global_step=step)

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print("[!] Restored from: {}".format(checkpoint))

    def generate(self, num_samples, start_string, num_classes):
        samples = list(start_string)
        # samples = [c for c in start_string]
        new_state = self.session.run(self.initial_state)
        preds = np.ones((num_classes, ))
        for c in start_string:
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.initial_state: new_state}
            preds, new_state = self.session.run([self.proba_prediction, self.final_state], feed_dict=feed) 
        c = pick_top_n(preds, num_classes)

        samples.append(c)

        for i in range(num_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.initial_state: new_state}

            preds, new_state = self.session.run([self.proba_prediction, self.final_state], feed_dict=feed)
            c = pick_top_n(preds, num_classes)
            samples.append(c)

        return samples