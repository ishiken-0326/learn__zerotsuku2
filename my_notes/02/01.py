text = 'You say goodbay and I say hello.'
print(text)
text = text.lower()
print(text)
text = text.replace('.', ' .')
print(text)

words = text.split(' ')
print(words)

word_to_id = {}
id_to_word = {}

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word

print(id_to_word)
print(word_to_id)

import numpy as np

corpus = [word_to_id[w] for w in words]
corpus = np.array(corpus)
print(corpus)


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    # words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    
    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word

corpus, word_to_id, id_to_word = preprocess(text)
print(f'courpus:{corpus}')
print(f'word_to_id:{word_to_id}')
print(f'id_to_word:{id_to_word}')


window_size = 1
vocab_size = len(word_to_id)
print(f'vocab_size: {vocab_size}')

corpus_size = len(corpus)
print(f'corpus_size: {corpus_size}')

print(word_to_id)

co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
print(co_matrix)
print(co_matrix.shape)

idx = 6
word_id = corpus[idx]
print(word_id)
print(id_to_word[word_id])

left_idx = idx - 1
print(left_idx)
right_idx = idx + 1
print(right_idx)

left_word_id = corpus[left_idx]
print(left_word_id)
print(id_to_word[left_word_id])
co_matrix[word_id, left_word_id] += 1
print(co_matrix)

right_word_id = corpus[right_idx]
print(right_word_id)
print(id_to_word[right_word_id])
co_matrix[word_id, right_word_id] += 1
print(co_matrix)

print(co_matrix[word_id])


def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)

    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - 1
            right_idx = idx + 1

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    
    return co_matrix

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
co_matrix = create_co_matrix(corpus, len(word_to_id), 1)
print(co_matrix)
print(co_matrix.shape)

print(word_to_id)


import matplotlib.pyplot as plt

def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)
    return np.dot(nx, ny)

# a_vec = np.array([5.0, 5.0])
# b_vec = np.array([3.0, 9.0])

# sim_val = cos_similarity(a_vec, b_vec)

# plt.quiver(0, 0, a_vec[0], a_vec[1], angles='xy', scale_units='xy', scale=1, color='c', label='a') # 有効グラフ
# plt.quiver(0, 0, b_vec[0], b_vec[1], angles='xy', scale_units='xy', scale=1, color='orange', label='b') # 有効グラフ
# plt.xlim(min(0, a_vec[0], b_vec[0]) - 1, max(0, a_vec[0], b_vec[0]) + 1)
# plt.ylim(min(0, a_vec[1], b_vec[1]) - 1, max(0, a_vec[1], b_vec[1]) + 1)
# plt.legend() # 凡例
# plt.grid() # 補助線
# plt.title('Similarity:' + str(np.round(sim_val, 3)), fontsize=20)
# plt.savefig('plot.jpg')

print(co_matrix)
print(np.sum(co_matrix))