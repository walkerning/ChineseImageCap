# -*- coding: utf-8 -*-

from __future__ import print_function

import sys

import tensorflow as tf

from im2txt.inference_utils.vocabulary import Vocabulary

if len(sys.argv) > 1:
    tfrecords_filename = sys.argv[1]
else:
    tfrecords_filename = "/home/foxfi/homework/pattern_recognition/project/code/tfrecord/train-00001.tfrecord"

filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=1)
reader = tf.TFRecordReader()
_, serialized = reader.read(filename_queue)
context, seq = tf.parse_single_sequence_example(
    serialized,
    context_features={
        "image/feature": tf.FixedLenFeature((4096,), dtype=tf.float32),
        "image/image_id": tf.FixedLenFeature((1,), dtype=tf.int64),
    },
    sequence_features={
        "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    })
images = context["image/feature"]
image_id = context["image/image_id"]
sequence = seq["image/caption_ids"]
# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

vocab = Vocabulary("../../vocab.txt")

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for _ in range(40):
        img, image_id_value, anno = sess.run([images, image_id, sequence])
        segs = [vocab.id_to_word(idx) for idx in anno]
        print("index {}, annotation: {}".format(image_id_value, "".join(segs)))
    coord.request_stop()
    coord.join(threads)
