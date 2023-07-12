import numpy as np

def softmax(x):
    if x.ndim == 2:
        x = x-x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    if x.ndim == 1:
        x = x-np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x

s = np.arange(-5, 5)
print(s)

y = softmax(s)
print(np.round(y, 3))
print(np.sum(y))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return - np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

k = 1
t = np.zeros_like(y)
t[k] = 1
print(t)

L = cross_entropy_error(y, t)
print(L)

class MatMul:

    def __init__(self, W):
        self.params = [W]
        self.grad = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out
    
    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)

        self.grad[0][...] = dW

lt = [np.array([0, 1, 2])]
lt


class Sigmoid:

    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))

        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
sigmod_layer = Sigmoid()

a = sigmod_layer.forward(y)
print(np.round(a, 3))
print(a.shape)
dout = np.random.randn(*a.shape)

da = sigmod_layer.backward(dout)
print(da.shape)