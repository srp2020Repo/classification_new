import tensorflow as tf
import datetime
import os
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


def get_tensorboard_callback(model_name):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join('data', 'autograph', model_name, stamp)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    return tensorboard_callback

def get_checkpoint_callback(checkpoint_save_path):
    cp_callback = ModelCheckpoint(filepath=checkpoint_save_path,
                                  save_weights_only=True,
                                  save_best_only=True)
    return cp_callback