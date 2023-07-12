import numpy as np
import sys
sys.path.append('../../deep-learning-from-scratch-2/')

from common.util import preprocess
from common.util import create_contexts_target

text = 'You say goodbye and I say hello.'

corpus, word_to_id, id_to_word = preprocess(text)
print(word_to_id)
print(id_to_word)
print(corpus)

vocab_size = len(word_to_id)
print(vocab_size)

contexts, target = create_contexts_target(corpus, window_size=1)
print(target)
print(target.shape)
print(contexts)
print(contexts.shape)

hidden_size = 3

W_in = np.random.randn(vocab_size, hidden_size)
print(np.round(W_in, 2))
print(W_in.shape)

h = W_in[contexts[:, 0]]
print(contexts[:, 0])
print(np.round(h, 2))
print(h.shape)

print('逆伝播')
dout = np.ones_like(h)
print(dout)
print(dout.shape)

dW_in = np.zeros_like(W_in)
print(dW_in)
print(dW_in.shape)

for i, word_id in enumerate(contexts[:, 0]):
    dW_in[word_id] += dout[i]
print(dW_in)

dW_in = np.zeros_like(W_in)
print(dW_in)

np.add.at(dW_in, contexts[:, 0], dout)
print(dW_in)


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
    
    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)

print('Embedding test')
print(dW_in)
embed_layer0 = Embedding(W_in)
embed_layer1 = Embedding(W_in)

h0 = embed_layer0.forward(contexts[:, 0])
h1 = embed_layer1.forward(contexts[:, 1])
h = (h0 + h1) * 0.5
print(np.round(h, 2))
print(h.shape)

dout = np.random.randn(contexts.shape[0], hidden_size)
print(dout.shape)

embed_layer0.backward(dout)
embed_layer1.backward(dout)

dW_in0, = embed_layer0.grads
dW_in1, = embed_layer1.grads
print(dW_in0)
print(dW_in1)