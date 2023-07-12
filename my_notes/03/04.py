# coding: utf-8
import sys
sys.path.append('../../deep-learning-from-scratch-2/')

from common.util import preprocess, create_contexts_target, convert_one_hot
from common.layers import MatMul,SoftmaxWithLoss

import numpy as np

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V = vocab_size
        H = hidden_size
        
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in
    
    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = 0.5 * (h0 + h1)
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss
    
    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None
    
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
print(word_to_id)
print(id_to_word)
print(corpus)

window_size = 1
vocab_size = len(word_to_id)
print(vocab_size)

contexts, target = create_contexts_target(corpus, window_size)
print(contexts)
print(contexts.shape)
print(target)
print(target.shape)

contexts = convert_one_hot(contexts, vocab_size)
target = convert_one_hot(target, vocab_size)
print(contexts)
print(contexts.shape)
print(target)
print(target.shape)

hidden_size = 5

model = SimpleCBOW(vocab_size, hidden_size)
print(np.round(model.params[0], 3))
print(np.round(model.params[1], 3))
print(np.round(model.word_vecs, 3))



from common.trainer import Trainer
from common.optimizer import Adam

hidden_size = 5
model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

batch_size = 3
max_epoch = 1000

trainer.fit(contexts, target, max_epoch, batch_size, eval_interval=2)
trainer.plot()

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])
