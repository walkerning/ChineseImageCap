# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from gensim.models import Word2Vec

from im2txt.inference_utils.vocabulary import Vocabulary

embedding_file = "embedding/zh.bin"
vocab_file = "../../vocab_nocut.txt"
vocab = Vocabulary(vocab_file)

embedding = Word2Vec.load(embedding_file)
embedding_list = []
for idx, word in enumerate(vocab.reverse_vocab):
    if idx not in {vocab.start_id, vocab.end_id, vocab.unk_id}:
        word = word.decode("utf-8")
        if word in embedding:
            embedding_list.append(embedding[word].astype(np.float32))
            continue
    embedding_list.append(np.random.uniform(-0.08, 0.08, (300,)).astype(np.float32))

np.array(embedding_list).tofile("embedding/zh.npz")
