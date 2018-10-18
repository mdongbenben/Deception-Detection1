import chainer 
import numpy as np
import chainer.function as F
from chainer import training
import gensim
chainer.functions.concat

#from chainer.training import extension 

# call pre-traind VGG16
from chainer.links import VGG16Layers
model_1 = VGG16Layers()
print (model_1)
#model_1 = F.flatten(model_1)

# call word2vec
from gensim.models import word2vec
data = word2vec.Text8Corpus('corpus80.txt')
#model = word2vec.Word2Vec(data, size=200)
model = word2vec.Word2Vec(data, size=300)
# model = F.flatten(model)

#out = model.most_similar(positive=['delicious', 'meal'])
#for x in out:
#    print(x[0], x[1])

# Concatenates two layers
new_layers = F.concat(model, model_2, axis=1)


# Full coneccted alyer


# Softmax


学習
# tra....

# opt

# vi



