import os, random

import tensorflow as tf

class SequenceData(object):
    """生成序列任务，每个序列有不同的长度。
    两类数据：
    - 类别0: 递增序列（如[0, 1, 2, 3, 4, ...]）
    - 类别1: 完全随机的序列（[1, 3, 10, 7, ...]）
    """
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
                max_value=1000):
        self.data, self.labels, self.seq_len = [], [], []

        for _ in range(n_samples):
            _len = random.randint(min_seq_len, max_seq_len)
            self.seq_len.append(_len)
            
            # random.random 生成0-1之间的随机数
            if random.random() < .5:
                # 递增序列
                rand_start = random.randint(0, max_value - _len)
                s = [[float(i)/max_value] for i in range(rand_start, rand_start+_len)]
                s += [[0.] for _ in range(max_seq_len - _len)]
                self.data.append(s)
                self.labels.append([1., 0.])
            else:
                s = [[float(random.randint(0, max_value) / max_value)] for _ in range(_len)]
                s += [[0.] for _ in range(max_seq_len - _len)]
                self.data.append(s)
                self.labels.append([0., 1.])
    
    def next(self, batch_size):
        batch_data, batch_labels, batch_seq_len = [], [], []
        for _ in range(batch_size):
            i = random.randint(0, len(self.data)-1)        
            batch_data.append(self.data[i])
            batch_labels.append(self.labels[i])
            batch_seq_len.append(self.seq_len[i])
        return batch_data, batch_labels, batch_seq_len


class Classifier:
    
    def __init__(self, learning_rate=0.01, max_steps=1000000, batch_size=128,
                    log_step=10, seq_len=20, hidden_size=164, num_classes=2, model_path=""):
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.log_step = log_step
        self.max_seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_classes = num_classes
   
        self.build_graph()

    def build_graph(self):
        input = tf.placeholder("float", [None, self.max_seq_len, 1])
        label = tf.placeholder("float", [None, self.num_classes])

        seq_lens = tf.placeholder(tf.int32, [None])

        weights = tf.Variable(tf.random_normal([self.hidden_size, self.num_classes]))
        bias = tf.Variable(tf.random_normal([self.num_classes]))

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)

        outputs, states = tf.nn.dynamic_rnn(lstm_cell, input, dtype=tf.float32,
                                                sequence_length=seq_lens)
        
        # 索引
        batch_size = tf.shape(outputs)[0]   # 获取真正的batch_size，因为测试集和训练集的batch_size不同。
        index = tf.range(0, batch_size,) * self.max_seq_len + (seq_lens - 1) 
        outputs = tf.gather(tf.reshape(outputs, [-1, self.hidden_size]), index)

        pred = tf.matmul(outputs, weights) + bias
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(cost)
        
        # 准确率
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
        accuarcy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()

        trainset = SequenceData(2000)
        testset = SequenceData(500)
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            while step * self.batch_size < self.max_steps:
                batch_x, batch_y, batch_seq_len = trainset.next(self.batch_size)
                _, acc, loss = sess.run([optimizer, accuarcy, cost], feed_dict={input: batch_x, label: batch_y, seq_lens: batch_seq_len})
                
                if step % self.log_step == 0:
                    print("Iter %s, Minibatch Loss= %.6f, Training Accuracy= %.5f" % (step*self.batch_size, loss, acc))
                step += 1
            print("==============Training Done===============>")
            self.saver = tf.train.Saver()
            self.saver.save(sess, os.path.join("./model/", "model"), global_step=step)
            
            print("Testing Accuracy: ", sess.run(accuarcy, 
                                                feed_dict={input: testset.data,
                                                            label: testset.labels,
                                                            seq_lens: testset.seq_len}))

if __name__ == "__main__":
    #trainset = SequenceData(5)
    #print(trainset.data, "\n", trainset.labels, "\n", trainset.seq_len)
    model = Classifier()