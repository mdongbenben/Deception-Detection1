import os
from data import *
import numpy as np

import tensorflow as tf
from model import Model
from config import Config


def get_model():
    # 初始化模型
    model = Model(config, vocab_size)
    return model


def feed_data(x, y1, y2):
    feed_dict = {
        model.input_x: x,
        model.arousal_Y: y1,
        model.valence_Y: y2
    }
    return feed_dict


def train():

    print("开始配置 TensorBoard ")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 准备数据
    print("开始读取数据文件")
    train_raw_data = read_csv_file(config.train_file_trans, config.train_file_video)
    train_data = sent_to_index(30, word2id, train_raw_data)

    dev_raw_data = read_csv_file(config.dev_file_trans, config.dev_file_video)
    dev_data = sent_to_index(30, word2id, dev_raw_data)
    print("读取数据完成\n")

    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: word_vectors})

    best_loss_val = 1.0
    for epoch in range(config.epoch_num):
        print("Running epoch {}".format(epoch+1))
        train_loss = 0.0

        for i, sample in enumerate(train_data):
            if len(sample[2]) > 0:
                X, arousal_Y, valence_Y = np.array([sample[2]]), np.array(sample[-2]), np.array(sample[-1])    
                
                feed_dict = feed_data(X, arousal_Y, valence_Y)

                s = sess.run(model.loss_summary, feed_dict=feed_dict)
                writer.add_summary(s, i)

                model_loss=sess.run(model.loss, feed_dict=feed_dict)
                
                print("[Sample {}] loss: {}".format(i+1, model_loss[0]))
                sess.run(model.optim, feed_dict=feed_dict)
                
                train_loss += model_loss[0][0]

        epoch_train_loss = train_loss / float(len(train_data))
        print("\n\n[Epoch {}] Average train loss: {}\n\n".format(epoch+1, epoch_train_loss))

        
        dev_loss = 0.0
        # 每训一个epoch，测试一次
        for i, dev_sam in enumerate(dev_data):
            if len(dev_sam[2]) > 0:
                X, arousal_Y, valence_Y = np.array([dev_sam[2]]), np.array(dev_sam[-2]), np.array(dev_sam[-1])

                loss = sess.run(model.loss, feed_dict=feed_dict)

                dev_loss += loss[0][0]

        epoch_dev_loss = dev_loss / float(len(dev_data))
        print("\n\n[Epoch {}] Average dev loss: {}\n\n".format(epoch+1, epoch_dev_loss))
        
        # 如果验证集上的loss下降，则把这个epoch训练的模型存下来
        if epoch_dev_loss < best_loss_val:
            best_loss_val = epoch_dev_loss
            saver.save(sess=sess, save_path=config.save_path)


if __name__ == '__main__':
    import sys
    flag = sys.argv[1]

    config = Config()

    # 准备word vector
    print("开始加载预训练的词向量")
    word_vectors, word2id = load_embed_dict(config.word_vector_path)
    with open(os.path.join(config.save_dir, "vocab.txt"), "w") as f:
         for k,v in word2id.items():
             f.write(k+"\n")
    # word_vectors, word2id = load_embed_glove("glove.twitter.27B.100d.txt")
    print("加载完成\n")

    vocab_size = len(word2id)

    # 准备模型
    print("开始构建模型")
    model = get_model()
    print("构建完成\n")

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    train()

            


