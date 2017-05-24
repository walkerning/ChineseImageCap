# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math

import h5py
import numpy as np
import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("dataset_name", "test_set", "can be one of train_set validation_set, test_set")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
# tf.flags.DEFINE_string("input_files", "",
#                        "File pattern or comma-separated list of file patterns "
#                        "of image files.")
tf.flags.DEFINE_string("feature_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_string("test_index_file", "", "a txt test index file")

tf.flags.DEFINE_string("write_top_res_file", None, "write top1 caption to file")
tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  feature_fnames = FLAGS.feature_files.split(",")
  with open(FLAGS.test_index_file, "r") as f:
    test_inds = [int(line.strip()) for line in f.readlines()]
  
    
  test_features = np.concatenate([h5py.File(feature_fname, "r")[FLAGS.dataset_name][test_inds] for feature_fname in feature_fnames], axis=1)
    
    
  # filenames = []
  # for file_pattern in FLAGS.input_files.split(","):
  #   filenames.extend(tf.gfile.Glob(file_pattern))
  # tf.logging.info("Running caption generation on %d files matching %s",
  #                 len(filenames), FLAGS.input_files)
  
  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)
    
    write_res_f = None
    if FLAGS.write_top_res_file is not None:
      write_res_f = open(FLAGS.write_top_res_file, "w")

    for ind, feature in zip(test_inds, test_features):
      captions = generator.beam_search(sess, feature)
      if write_res_f is not None:
        sentence = [vocab.id_to_word(w) for w in captions[0].sentence[1:-1]]
        print(" ".join(sentence), file=write_res_f)
      print("Captions for test image feature %d:" % ind)
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))


if __name__ == "__main__":
  tf.app.run()
