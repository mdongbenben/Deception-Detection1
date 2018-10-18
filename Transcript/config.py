import os

class Config(object):
    """神经网络配置参数"""

    # 模型参数
    embedding_dim = 300      # 词向量维度
    seq_length = 30        # 序列长度

    blstm_layers= 1           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元

    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    attention_da = 350      # attn参数
    attention_r = 30        # attn参数

    epoch_num = 100         # 总迭代轮次

    word_vector_path = "GoogleNews-vectors-negative300.bin"    # 训练好的词向量文件路径

    train_file_trans = "ziqi_text/omg_TrainTranscripts.csv"    # train文件路径
    train_file_video = "ziqi_text/omg_TrainVideos.csv"
        
    dev_file_trans = "ziqi_text/omg_ValidationTranscripts.csv" # dev文件路径
    dev_file_video = "ziqi_text/omg_ValidationVideos.csv"

    save_dir = "checkpoints"
    save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证模型结果保存路径