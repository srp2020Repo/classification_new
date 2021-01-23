import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import DenseNet121
from configuration import NUM_CHANNELS

def build_DenseNet121(IMG_SIZE=224, channels=3, trainable_layers=10):
    base = DenseNet121(weights=None,
                    include_top=False,
                    input_shape=(IMG_SIZE, IMG_SIZE, channels))
    base.trainable = True
    '''
    for layer in base.layers[:-trainable_layers]:
        layer.trainable = False
    for layer in base.layers[-trainable_layers:]:
        layer.trainable = True
    '''
    model = models.Sequential()
    model.add(base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    return model

build_DenseNet121()