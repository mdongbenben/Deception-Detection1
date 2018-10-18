import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple,DropoutWrapper
from tensorflow.contrib.layers import fully_connected

class Model():
    def __init__(self, config, vocab_size):
        """独白文本情感分类任务，采用Bi-LSTM加self-Attention模型。
           神经网络，输入为文本索引列表，经过embeding层，attention层，
           将hidden layer的结果经过全连接层，map到aroural和valence两个值，
           范围分别为[0,1],[-1,1]。
        """
        self.config = config
        self.vocab_size = vocab_size

        # 模型输入参数
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.config.seq_length], name="input_x")
        self.arousal_Y = tf.placeholder(tf.float32, name="arousal_Y")
        self.valence_Y = tf.placeholder(tf.float32, name="valence_Y")
        
        # self-attention层的参数
        self.W_s1 = tf.Variable(tf.random_uniform([self.config.attention_da, self.config.hidden_dim * 2], -1, 1),
                            dtype=tf.float32, name="W_s1")

        self.W_s2 = tf.Variable(tf.random_uniform([self.config.attention_r, self.config.attention_da], -1, 1),
                            dtype=tf.float32, name="W_s2")

        self.batch_size = tf.shape(self.input_x)[0]
        self.build()

    def build(self):
        """构建模型"""

        with tf.device('/cpu:0'):
            W = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.config.embedding_dim]),
                             trainable=False, name="W")

            self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.config.embedding_dim])         
            self.embedding_init = W.assign(self.embedding_placeholder)
            self.embedding_inputs = tf.nn.embedding_lookup(W, self.input_x)

        with tf.name_scope("blstm"):
            # 多层bilstm网络
            lstm_f_cell = DropoutWrapper(LSTMCell(self.config.hidden_dim),
                                    output_keep_prob=self.config.dropout_keep_prob)
            lstm_b_cell = DropoutWrapper(LSTMCell(self.config.hidden_dim),
                                    output_keep_prob=self.config.dropout_keep_prob)
            outputs, _ = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_f_cell,
                                            cell_bw=lstm_b_cell,
                                            inputs=self.embedding_inputs,
                                            dtype=tf.float32, time_major=True)

            lstm_output = tf.concat([outputs[0][0], outputs[1][0]], axis=1)

        with tf.name_scope("self_attn"):
            linear_out = tf.nn.tanh(tf.matmul(self.W_s1, tf.transpose(lstm_output)))
            alpha = tf.nn.softmax(tf.matmul(self.W_s2, linear_out))
            attn_hidden_emb = tf.matmul(alpha, lstm_output)

            self.avg_attn_hidden_emb = tf.reshape(tf.div(tf.reduce_sum(attn_hidden_emb, 0), 
                                self.config.attention_r),[self.config.hidden_dim*2, 1])

        with tf.name_scope("loss"):
            # 全连接层，分别连接aroural和valence
            fc1 = fully_connected(self.avg_attn_hidden_emb, num_outputs=1, activation_fn=None)
            fc = tf.contrib.layers.dropout(fc1, self.config.dropout_keep_prob)

            self.arousal = tf.nn.sigmoid(fc)
            self.valence = tf.nn.tanh(fc)
        
            self.loss = (tf.pow(tf.subtract(self.arousal, self.arousal_Y), 2) + 
                    tf.pow(tf.subtract(self.valence, self.valence_Y), 2)) / float(2)
            
            self.loss_summary = tf.summary.scalar('loss', self.loss[0][0])

        with tf.name_scope("optimize"):
            # 优化
            self.optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)    
                