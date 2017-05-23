# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import math
import random
import argparse
from collections import defaultdict

import numpy as np
import h5py
import jieba
import tensorflow as tf

class Vocab(object):
    def __init__(self, start_word="<S>", end_word="</S>"):
        self.vocab_dict = {}
        self.vocab_lst = []
        self.occur_dict = {}
        self.last_index = 0
        self.start_id = self.add_to_vocab(start_word)
        self.end_id = self.add_to_vocab(end_word)

    @property
    def vocab_size(self):
        return len(self.vocab_lst)

    def dump_to_file(self, fname):
        with open(fname, "w") as f:
            for word in self.vocab_lst:
                print("{} {}".format(word.encode("utf-8"), self.occur_dict[word]), file=f)
            
    def word_to_id(self, word):
        return self.vocab_dict[word]

    def id_to_word(self, id_):
        return self.vocab_lst[id_]

    def add_to_vocab(self, word):
        if word in self.vocab_dict:
            self.occur_dict[word] = self.occur_dict[word] + 1
        else:
            self.vocab_dict[word] = self.last_index
            self.vocab_lst.append(word)
            self.occur_dict[word] = 1
            self.last_index += 1
        return self.vocab_dict[word]

def parse_caption_file(fname, vocab, seg_dict, cut):
    indexes = []
    index = None
    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                index = int(line)
                indexes.append(index)
            except ValueError:
                if cut:
                    seg_list = list(jieba.cut(line))
                else:
                    seg_list = list(line)
                seg_dict[index].append([vocab.start_id] +
                                       [vocab.add_to_vocab(word) for word in seg_list] +
                                       [vocab.end_id])
    return indexes

def parse_image_feature_file(fnames):
    train_set = None
    val_set = None
    test_set = None
    for fname in fnames:
        dataset = h5py.File(fname, "r")
        assert set(dataset.keys()) == {u"test_set", u"train_set", u"validation_set"}
        if train_set is None:
            train_set, val_set, test_set = dataset["train_set"], dataset["validation_set"], dataset["test_set"]
        else:
            train_set, val_set, test_set = np.hstack((train_set, dataset["train_set"])), np.hstack((val_set, dataset["validation_set"])), np.hstack((test_set, dataset["test_set"])), 
    return train_set, val_set, test_set

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature_list(values):
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def to_example(image_id, feature, seg):
    context = tf.train.Features(feature={
        "image/image_id": _int64_feature(image_id),
        "image/feature": _float_feature(feature.tolist())
    })
    feature_lists = tf.train.FeatureLists(feature_list={
        "image/caption_ids": _int64_feature_list(seg)
    })
    example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)
    return example

def dump_tfrecord(indexes, image_features, seg_dict, tfr_dir, num_samples_per_record, split):
    # dump tfrecord files
    print("dump train tfrecords")
    samples = [(ind, image_features[i], seg) for i, ind in enumerate(indexes) for seg in seg_dict[ind]]
    print("{}: num of samples: {}".format(split, len(samples)))
    random.seed(12345)
    random.shuffle(samples)
    f_idx = 1
    f_num = 0
    fname = os.path.join(tfr_dir, "%s-%.5d.tfrecord"%(split, f_idx))
    print("writing {} records to {}".format(split, fname))
    writer = tf.python_io.TFRecordWriter(fname)
    for sample in samples:
        example = to_example(*sample)
        writer.write(example.SerializeToString())
        f_num += 1
        if f_num == num_samples_per_record:
            f_idx += 1
            f_num = 0
            fname = os.path.join(tfr_dir, "%s-%.5d.tfrecord"%(split, f_idx))
            print("writing {} records to {}".format(split, fname))
            writer = tf.python_io.TFRecordWriter(fname)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("val_file")
    parser.add_argument("vocab_file", help="dump vocabulary to file")
    parser.add_argument("tfrecord_dir", help="directory of tfrecord")
    parser.add_argument("--num-samples-per-record", default=400, type=int)
    parser.add_argument("--vocab-only", default=False, action="store_true")
    parser.add_argument("--no-cut", default=False, action="store_true")

    # TODO: multiple image feature file
    parser.add_argument("-f", "--image-feature-file", action="append", required=True,
                        help="HDF5 file contains the CNN-extracted features for each image id")
    
    args = parser.parse_args()

    vocab = Vocab()
    seg_dict = defaultdict(lambda : [])
    print("parsing caption from train: {}, val: {}".format(args.train_file, args.val_file))
    train_indexes = parse_caption_file(args.train_file, vocab, seg_dict, not args.no_cut)
    val_indexes = parse_caption_file(args.val_file, vocab, seg_dict, not args.no_cut)
    print("num train indexes: {}; num val indexes: {}".format(len(train_indexes), len(val_indexes)))
    # dump vocab file
    print("vocabulary size: {}; dump vocabulary to {}".format(vocab.vocab_size, args.vocab_file))
    vocab.dump_to_file(args.vocab_file)
    if args.vocab_only:
        return

    # load CNN-extracted image features
    print("loading features from {}".format(args.image_feature_file))
    train_features, val_features, _ = parse_image_feature_file(args.image_feature_file)
    assert len(train_indexes) == train_features.shape[0]
    assert len(val_indexes) == val_features.shape[0]

    # TODO: test_feature自己读并且喂placeholder...就不dump了
    dump_tfrecord(train_indexes, train_features, seg_dict, args.tfrecord_dir, args.num_samples_per_record, "train")
    dump_tfrecord(val_indexes, val_features, seg_dict, args.tfrecord_dir, args.num_samples_per_record, "val")

if __name__ == "__main__":
    main()
