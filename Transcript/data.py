import re
import math
import random
import numpy as np
import pandas as pd

from collections import defaultdict

from gensim.models import KeyedVectors


def load_embed_dict(embed_vec_path, bina=True):
    """ 加载预训练的词向量
    """
    word_vec, word2id = [], {}
    word2vec = KeyedVectors.load_word2vec_format(embed_vec_path, binary=bina)
    for ix, token in enumerate(word2vec.vocab):
        word_vec.append(word2vec[token])
        word2id[token] = ix
    return np.array(word_vec,dtype='f'), word2id


def load_embed_glove(embed_vec_path):
    with open(embed_vec_path, "r") as f:
        word_vec, word2id = [], {}
        for ix, line in enumerate(f.readlines()):
            tokens = line.strip().split(" ")
            word_vec.append([float(token) for token in tokens[1:]])
            word2id[tokens[0]] = ix
    return np.array(word_vec,dtype='f'), word2id


def read_csv_file(csv_trans_path, csv_video_path):
    """ 读取文件
    """
    id_utterance = pd.read_csv(csv_trans_path,delimiter=",").drop(["link"], axis=1).fillna("")
    arousal_valence = pd.read_csv(csv_video_path,delimiter=",").drop(["link", "start", "end", "EmotionMaxVote"], axis=1)
    df = id_utterance.merge(arousal_valence, how='left')
    df["video"] = df["video"].str.split("_").str[0]
    df["utterance"] = df["utterance"].str.replace(".mp4","")
    raw_data = df.values.tolist()
    return raw_data


def sent_to_index(fix_length, word2id, raw_data):
    """ 将raw_data里的台词项变为index

        raw_data = [[video, utterance(str), trans, aroural, valence], [...], [...]]
        -> raw_data = [[video, utterance(int), trans, aroural, valence], [...], [...]]
    """
    for sample in raw_data:
        trans = abbr_to_com(sample[2].strip())
        tokens = trans.split()
        if len(tokens) > 0:
            if len(tokens) < fix_length:
                sample[2] = [word2id[t] if t in word2id else 0 for t in tokens] + ([0] * (fix_length-len(tokens)))
            else:
                sample[2] = [word2id[t] if t in word2id else 0 for t in tokens[:fix_length]]
        else:
            sample[2] = []
    return raw_data


def get_batch(batch_size, data):
    random.shuffle(data)
    sindex = 0
    eindex = batch_size
    while eindex < len(data):
        batch = data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch


def abbr_to_com(new_text):
    # to find the 's following the pronouns. re.I is refers to ignore case
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    # to find the 's following the letters
    pat_s = re.compile("(?<=[a-zA-Z])\'s")
    # to find the ' following the words ending by s
    pat_s2 = re.compile("(?<=s)\'s?")
    # to find the abbreviation of not
    pat_not = re.compile("(?<=[a-zA-Z])n\'t")
    # to find the abbreviation of would
    pat_would = re.compile("(?<=[a-zA-Z])\'d")
    # to find the abbreviation of will
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")
    # to find the abbreviation of am
    pat_am = re.compile("(?<=[I|i])\'m")
    # to find the abbreviation of are
    pat_are = re.compile("(?<=[a-zA-Z])\'re")
    # to find the abbreviation of have
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")

    new_text = pat_is.sub(r"\1 is", new_text)
    new_text = pat_s.sub("", new_text)
    new_text = pat_s2.sub("", new_text)
    new_text = pat_not.sub(" not", new_text)
    new_text = pat_would.sub(" would", new_text)
    new_text = pat_will.sub(" will", new_text)
    new_text = pat_am.sub(" am", new_text)
    new_text = pat_are.sub(" are", new_text)
    new_text = pat_ve.sub(" have", new_text)
    new_text = new_text.replace('\'', ' ')
    new_text = new_text.replace("f******", "fucking").replace("s***","shit").replace("a******","asshole")
    new_text = new_text.replace("b******", "bastard").replace("c***","crap")

    return new_text
    

# def statistics(raw_data):
#     len_num = defaultdict(int)
#     for sample in raw_data:
#         tokens = sample[0].strip().split()
#         if len(tokens) in len_num:
#             len_num[len(tokens)] += 1
#         else:
#             len_num[len(tokens)] = 1
#     print(sorted(len_num.items(), key=lambda x: x[0]))

# raw_data = read_csv_file("ziqi_text/omg_TrainTranscripts.csv", "ziqi_text/omg_TrainVideos.csv")
# statistics(raw_data)