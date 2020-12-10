"""This file contains some utility functions"""

import tensorflow as tf
import time
import os
FLAGS = tf.app.flags.FLAGS

def get_config():
  """Returns config for tf.session"""
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  return config

# def load_ckpt(saver, sess, ckpt_dir="train"):
#   """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
#   while True:
#     try:
#       latest_filename = "checkpoint_best" if ckpt_dir=="eval" else None
#       ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
#       ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
#       tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
#       saver.restore(sess, ckpt_state.model_checkpoint_path)
#       return ckpt_state.model_checkpoint_path
#     except:
#       tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
#       time.sleep(10)

def load_ckpt(saver, sess, load_best=False):
  """Load checkpoint from the train directory and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
  while True:
    try:
      ckpt_state = None
      if load_best:
        eval_dir = os.path.join(FLAGS.log_root, "eval")
        if os.path.exists(eval_dir):
          try:
            ckpt_state = tf.train.get_checkpoint_state(eval_dir, latest_filename="checkpoint_best")
          except ValueError:
            pass

      train_dir = os.path.join(FLAGS.log_root, "train")
      # ckpt_state = tf.train.get_checkpoint_state(train_dir)
      if ckpt_state is None:
        ckpt_state = tf.train.get_checkpoint_state(train_dir)

      tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
      saver.restore(sess, ckpt_state.model_checkpoint_path)
      return ckpt_state.model_checkpoint_path
    except:
        tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
        time.sleep(10)