import sys
sys.path.append('/home/user/projects/learn/zerotsuku2/deep-learning-from-scratch-2')

# スパイラル・データセット読み込み関数をインポート
from dataset import spiral

# その他この節で利用するライブラリ
import numpy as np
import matplotlib.pyplot as plt

x, t = spiral.load_data()
print(x.shape)
print(t.shape)

print(x[96:105])
print(t[96:105])

# 各クラスのデータ数
N = 100

# クラス数
class_num = 3

# 各クラスのマーカーを指定
markers = ['o', 'x', '^']

# 作図
for i in range(class_num):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], marker=markers[i]) # 散布図
plt.title('Spiral Data Set', fontsize=20) # タイトル
plt.show() # 描画