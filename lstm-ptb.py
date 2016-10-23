import argparse
import numpy as np
import sys
import tensorflow as tf
from reader import ptb_raw_data, ptb_iterator

class LstmPtb():
    def __init__(self, args):
        self.args = args
        self.initModel()
        
    def initModel(self):
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            self.input = tf.placeholder(tf.int32, [None, self.args.unfold_size], 'input')
            self.label = tf.placeholder(tf.int32, [None, self.args.unfold_size], 'label')
            softmax_W = tf.get_variable("softmax_W", shape=[self.args.hidden_size, self.args.vocab_size], dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b", shape=[self.args.vocab_size], dtype=tf.float32)
            
            embedding = tf.get_variable("embedding", shape=[self.args.vocab_size, self.args.embedding_size], dtype=tf.float32)
            embedding_input = tf.nn.embedding_lookup(embedding, self.input)
            
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.args.hidden_size, state_is_tuple=True)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.args.layer_no, state_is_tuple=True)
            self.initial_state = cell.zero_state(self.args.batch_size, tf.float32)
            state = self.initial_state
            outputs = []
            for i in range(self.args.unfold_size):
                if i > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(embedding_input[:, i, :], state)
                outputs.append(cell_output)
            self.final_state = state
    
            outputs1 = tf.concat(1, outputs)
            outputs2 = tf.reshape(outputs1, [-1, self.args.hidden_size])
            
            self.output = tf.matmul(outputs2, softmax_W) + softmax_b
            loss = tf.nn.seq2seq.sequence_loss_by_example(
                            [self.output], 
                            [tf.reshape(self.label, [-1])], 
                            [tf.ones([self.args.batch_size * self.args.unfold_size], dtype=tf.float32)]
                        )
                
            self.cost = tf.reduce_sum(loss) / self.args.batch_size            
            tvars = tf.trainable_variables()     
            grads, vars = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
            optimizer = tf.train.GradientDescentOptimizer(1.0)
            self.train_step = optimizer.apply_gradients(zip(grads, tvars))
            
            #self.train_step = optimizer.minimize(self.cost)
        
    def test(self, sess, word_to_id, words):
        state = sess.run(self.initial_state)
        start_word = 'we'
        word_id = word_to_id[start_word]
        word_no = 40
        
        print 'generating a sentense'
        sys.stdout.write ('%s ' % start_word)
        
        for i in range(word_no):
            output, state = sess.run([self.output, self.final_state], feed_dict= {
                        self.input : np.reshape([word_id] * self.args.batch_size, (self.args.batch_size, 1)),
                        self.initial_state[0].c : state[0].c,
                        self.initial_state[0].h : state[0].h,
                        self.initial_state[1].c : state[1].c,
                        self.initial_state[1].h : state[1].h,
            })
            word_id = np.argmax(output[0])
            word = words[word_id]
            sys.stdout.write ('%s ' % word)
        
    def gogo(self):
        print 'start training'
        
        train_data, valid_data, test_data, vocabulary, word_to_id, words = ptb_raw_data('simple-examples/data')
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(self.args.max_epoch):
            print 'running epoch %s' % (epoch + 1)
            step = 0
            state = sess.run(self.initial_state)
            for (x, y) in ptb_iterator(train_data, self.args.batch_size, self.args.unfold_size):
                a, cost, state = sess.run([self.train_step, self.cost, self.final_state], feed_dict={
                        self.input : x,
                        self.label : y,
                        self.initial_state[0].c : state[0].c,
                        self.initial_state[0].h : state[0].h,
                        self.initial_state[1].c : state[1].c,
                        self.initial_state[1].h : state[1].h,
                    })
                step += 1
                if step % 1000 == 0:
                    print 'cost : %.2f' % cost
                    # DJDJ
                    break
                
            self.test(sess, word_to_id, words)
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--batch-size', nargs='*', type=int, default=30)
    #parser.add_argument('--unfold-size', nargs='*', type=int, default=20)
    parser.add_argument('--batch-size', nargs='*', type=int, default=30)
    parser.add_argument('--unfold-size', nargs='*', type=int, default=1)
    parser.add_argument('--hidden-size', nargs='*', type=int, default=200)
    parser.add_argument('--embedding-size', nargs='*', type=int, default=100)
    parser.add_argument('--vocab-size', nargs='*', type=int, default=10000)
    parser.add_argument('--layer-no', nargs='*', type=int, default=2)
    parser.add_argument('--max_epoch', nargs='*', type=int, default=1000)
    args = parser.parse_args()
    
    LstmPtb(args).gogo()