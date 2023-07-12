import numpy as np
import sys
sys.path.append('../../deep-learning-from-scratch-2/')

from common.layers import MatMul
# コンテキストデータを指定
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]]) # you
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]]) # goodbye

# 重みの初期値をランダムに生成
W_in = np.random.randn(7, 3)  # 入力層
W_out = np.random.randn(3, 7) # 出力層

in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)

h = 0.5 * (h0 + h1)
print(h0)
print(h1)
print(h)

s = out_layer.forward(h)

print(np.round(s, 3))