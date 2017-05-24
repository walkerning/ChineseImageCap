# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np
from im2txt.inference_utils.vocabulary import Vocabulary
from im2txt.configuration import ModelConfig
from scipy import spatial
import cPickle

# sys.argv[1] : vocab file
# sys.argv[2] : check point file
# sys.argv[3]: store pkl file

class Embeddings(object):
    def __init__(self, fname=None, vocab_fname=None):
        if fname is not None:
            self.embeddings = cPickle.load(open(fname, "r"))
        if vocab_fname is not None:
            self.vocab = Vocabulary(vocab_fname)
            self.word_to_id = self.vocab.word_to_id
            self.id_to_word = self.vocab.id_to_word

    @classmethod
    def init_from_data(cls, embeddings, vocab):
        self = Embeddings()
        self.embeddings = embeddings
        self.vocab = vocab
        self.word_to_id = self.vocab.word_to_id
        self.id_to_word = self.vocab.id_to_word
        return self
    
    def most_similar(self, word, top_k=5):
        if not isinstance(word, np.ndarray):
            ind = self.vocab.vocab[word]
            emb = self.embeddings[ind]
        else:
            ind = None
            emb = word
        similarites = [(1-spatial.distance.cosine(emb, other), i) for i, other in enumerate(self.embeddings) if i != ind]
        return [self.vocab.id_to_word(ans[1]) for ans in sorted(similarites, key=lambda y: y[0], reverse=True)[:top_k]]

    def word_to_emb(self, word):
        return self.embeddings[self.vocab.vocab[word]]

    def most_similar_relation(self, r1, r2, o1, top_k=5):
        return self.most_similar(self.word_to_emb(r2) - self.word_to_emb(r1) + self.word_to_emb(o1), top_k)

test_cases = ["男人", "登山", "快艇", "树上", "一座", "鸟", "蛋糕", "行走", "列车"]
test_relation_cases = [("列车", "行驶", "狗"), ("列车", "行驶", "人"), ("列车", "行驶", "鸟")]

def main():
    vocab = Vocabulary(sys.argv[1])
    config = ModelConfig()
    ids, words = zip(*enumerate(vocab.reverse_vocab))
    
    seqs = tf.placeholder(tf.int64)
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
        embedding_map = tf.get_variable(
            name="map",
            shape=[config.vocab_size, config.embedding_size])
        seq_embeddings = tf.nn.embedding_lookup(embedding_map, seqs)
    saver = tf.train.Saver([embedding_map])
    with tf.Session() as sess:
        saver.restore(sess, sys.argv[2])
        embeddings = sess.run(seq_embeddings, feed_dict={
            seqs: np.array(ids[:-1]) # do not need unk_word
        })
        with open(sys.argv[3], "w") as f:
            cPickle.dump(embeddings, f)
    e = Embeddings.init_from_data(embeddings, vocab)
    for test_case in test_cases:
        print("与 `{}` 最接近的词: ".format(test_case))
        [print(x, end=", ") for x in e.most_similar(test_case)]
        print("\n----")
    for test_relation_case in test_relation_cases:
        print("`{}` -> `{}`; `{}` -> 最接近的词: ".format(*test_relation_case))
        [print(x, end=", ") for x in e.most_similar_relation(*test_relation_case)]
        print("\n----")
    l2norms = np.sum(e.embeddings**2, axis=1)    
    sorted_index = np.argsort(l2norms)[::-1]
    for i in range(30):
        idx = sorted_index[i]
        print("#{}: word: {}; occur time: {}; l2 norm: {}".format(i, e.id_to_word(idx), e.vocab.occur_time[idx], l2norms[idx]))

if __name__ == "__main__":
    main()
