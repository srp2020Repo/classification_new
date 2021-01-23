import tensorflow as tf
import numpy as np
from configuration import NUM_EPOCHS, BATCH_SIZE, model_name, checkpoint_save_path, figure_path

from models.DenseNet.dense121 import build_DenseNet121

from utils.load_img import load_img
from utils.callbacks import get_tensorboard_callback, get_checkpoint_callback
from utils.plot_metrics import plot_metric



ds_train = tf.data.Dataset.list_files('./dataset/train/*.png') \
                          .map(load_img, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                          .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
                          .prefetch(tf.data.experimental.AUTOTUNE)

ds_valid = tf.data.Dataset.list_files('./dataset/valid/*.png') \
                          .map(load_img, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                          .batch(BATCH_SIZE) \
                          .prefetch(tf.data.experimental.AUTOTUNE)



model = build_DenseNet121()
model.summary()
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2),
    loss = tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

#callbacks
tensorboard_callback = get_tensorboard_callback(model_name)
cp_callback = get_checkpoint_callback(checkpoint_save_path)

history = model.fit(
    ds_train,
    epochs=NUM_EPOCHS,
    validation_data=ds_valid,
    callbacks = [tensorboard_callback, cp_callback],
    workers=4
)

plot_metric(history, figure_path)