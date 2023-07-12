import numpy as np

c = np.array([[1, 0, 0, 0, 0, 0, 0]])
print(c.shape)

W = np.random.randn(7, 3)
print(W)

h = np.dot(c, W)
print(h)
print(h.shape)
