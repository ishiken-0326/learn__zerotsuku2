import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from PIL import Image

img = Image.open('./31909713-d9046856-b7ef-11e7-98fe-8a1e133c0010.png').convert('L').resize((100,100))
X = np.asarray(img)
print(X)
print(X.shape)
U, Sigma, VT = linalg.svd(X, full_matrices=True)

print(f'X:{X.shape}, U:{U.shape}, Σ:{Sigma.shape}, V^T:{VT.shape}')

total = np.zeros((100, 100))

for rank in [1, 2, 3, 4, 5]:
    # rank番目までの要素を抽出
    U_i = U[:, :rank]
    Sigma_i = np.matrix(linalg.diagsvd(Sigma[:rank], rank, rank))
    VT_i = VT[:rank, :]

    # rank番目の特異値以外をすべて０にして、rank番目の要素のみ残す
    if rank > 1:
        for ri in range(rank - 1):
            Sigma_i[ri, ri] = 0

    # 画像を復元
    temp_image = np.asarray(U_i * Sigma_i * VT_i)
    Image.fromarray(np.uint8(temp_image))

    # rank番目の要素のみを足す
    total += temp_image

    # rank番目までの要素で復元した画像と、行列Vのrank列目の値のプロットを比較してみる
    plt.figure(figsize=(5, 5))
    plt.suptitle(f"$u_{rank}$")
    plt.subplot(211)
    plt.imshow(temp_image, cmap="gray")
    plt.subplot(212)
    plt.plot(VT[0])
    plt.savefig(f'rank_{rank}.jpg')