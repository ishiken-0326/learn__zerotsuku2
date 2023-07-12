# coding: utf-8
import sys
sys.path.append('../../deep-learning-from-scratch-2/')

from common.util import preprocess
import numpy as np

text = 'You say goodbye and I say hello.'

corpus, word_to_id, id_to_word = preprocess(text)
print(word_to_id)
print(id_to_word)
print(corpus)

window_size = 1


def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]

    contexts = []

    for idx in range(window_size, len(corpus) - window_size):

        cs = []

        for t in range(-window_size, window_size+1):

            if t == 0:
                continue

            cs.append(corpus[idx+t])
        
        contexts.append(cs)

    return np.array(contexts), np.array(target)


# one-hot表現への変換関数の実装
def convert_one_hot(corpus, vocab_size):
    
    # ターゲットの単語数を取得
    N = corpus.shape[0]
    
    # one-hot表現に変換
    if corpus.ndim == 1: # 1次元配列のとき
        
        # 変換後の形状の2次元配列を作成
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        
        # 単語ID番目の要素を1に置換
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
    
    elif corpus.ndim == 2: # 2次元配列のとき
        
        # コンテキストサイズを取得
        C = corpus.shape[1]
        
        # 変換後の形状の3次元配列を作成
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        
        # 単語ID番目の要素を1に置換
        for idx_0, word_ids in enumerate(corpus): # 0次元方向
            for idx_1, word_id in enumerate(word_ids): # 1次元方向
                one_hot[idx_0, idx_1, word_id] = 1
    
    return one_hot


contexts, target = create_contexts_target(corpus, window_size=1)
print(f'contexts:{contexts}')
print(f'target:{target}')
print(target.shape)

vocab_size = len(word_to_id)
target_one_hot = convert_one_hot(target, vocab_size)
print(target)
print(target.shape)
print(target_one_hot)
print(target_one_hot.shape)

contexts_one_hot = convert_one_hot(contexts, vocab_size)
print(contexts)
print(contexts.shape)
print(contexts_one_hot)
print(contexts_one_hot.shape)

print(contexts_one_hot[:, 0])
print(contexts_one_hot[:, 1])