import tensorflow as tf
import  numpy as np
import layers

class Model(object):

    def __init__(self,params):
        self.input_word_dim = params['input_word_dim']
        self.s_max_len = params['s_max_len']
        self.q_max_len = params['q_max_len']
        self.depth = params['depth']
        self.L = params['L'] #lstm dim
        self.K = params['K'] #information dim
        self.discount = params['discount']

    def build_graph(self):
        self.input_q = tf.placeholder(tf.float32,[1,self.q_max_len,self.input_word_dim])
        self.input_q_len = tf.placeholder(tf.int32,[1])
        self.input_s = tf.placeholder(tf.float32,[None,self.s_max_len,self.input_word_dim])
        self.input_s_len = tf.placeholder(tf.int32, [None])

        self.y = tf.placeholder(tf.float32, [None])
        # self.last_x = tf.placeholder(tf.float32,[self.L,1])
        # self.input_h = tf.placeholder(tf.float32, [self.K, 1])

        self.Vq = tf.Variable(tf.truncated_normal(shape=[self.K, self.L], stddev=5e-2), name='Vq')
        self.V = tf.Variable(tf.truncated_normal(shape=[self.K, self.L], stddev=5e-2), name='V')
        self.W = tf.Variable(tf.truncated_normal(shape=[self.K, self.K], stddev=5e-2), name='W')
        self.U = tf.Variable(tf.truncated_normal(shape=[self.L, self.K], stddev=5e-2), name='U')

        q_lstm_output, q_lstm_state = self.dynamic_lstm_layer(self.input_q, self.L, 'q_lstm',
                                                                       input_len=self.input_q_len)
        q_last_state = q_lstm_state[1]
        self.q_last_state = tf.reshape(q_last_state,[self.L,1])

        s_lstm_output, s_lstm_state = self.dynamic_lstm_layer(self.input_s, self.L, 's_lstm',
                                                                       input_len=self.input_s_len)
        # s_last_state = s_lstm_state[1]

        s_last_state, first_attention_score = layers.matrix_attention_layer(s_lstm_output, q_last_state, self.L, 'v_global_attention')

        self.h0 = tf.matmul(self.Vq, self.q_last_state)
        self.h_now = tf.nn.sigmoid(self.h0) # K * 1
        # self.h_now = tf.nn.sigmoid(tf.matmul(self.V,self.last_x)+tf.matmul(self.W,self.input_h) + self.h0)

        self.prob = tf.matmul(tf.matmul(s_last_state,self.U),self.h_now) # None * 1
        self.prob = tf.reshape(self.prob,[-1])
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.prob, name='cross_entropy')
        # self.values, self.indices = tf.nn.top_k(tf.reshape(self.prob, [-1]), k=1, sorted=True, name='top_k')
        # self.chose = self.indices[0]
        # self.chose_s = s_last_state[self.chose]
        # self.chose_r = self.y[self.chose]
        # self.prob = tf.reshape(self.prob,[1,-1])
        # self.softmax = tf.nn.softmax(self.prob,dim = 1)
        # self.loss = -1*self.chose_r*tf.log(self.softmax[0,self.chose])

        # self.loss = tf.losses.softmax_cross_entropy(tf.reshape(self.y,[1,-1]),tf.reshape(self.prob,[1,-1]))
        tf.summary.scalar('cross entropy', self.loss)



    def dynamic_lstm_layer(self, input, hidden_dim, scope_name, input_len=None):
        with tf.variable_scope(scope_name):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim)
            lstm_output, final_state = tf.nn.dynamic_rnn(lstm_cell, input, sequence_length=input_len,
                                                         dtype=tf.float32, scope=scope_name)
            return lstm_output, final_state