import tensorflow as tf
import numpy as np
from configuration import CHANNELS, NUM_CHANNELS

def load_img(img_path, channels=CHANNELS):
    '''
    channels:
        'full' for 4 channels('rgb'+'mask')
        'rgb' for 3 channels('rgb')
        'mask' for 1 channel('mask')
    '''
    label = tf.constant(1,tf.int8) if tf.strings.regex_full_match(img_path,".*gctb.*") \
            else tf.constant(0,tf.int8)
    
    img = np.load(img_path.numpy())
    img = tf.convert_to_tensor(img, tf.float32)
   
    if channels == 'rgb':
        img = img[:,:,:3]
    elif channels == 'mask':
        img = img[:,:,3:4]
    return (img,label)

def wrap_function(img_path, NUM_CHANNELS=NUM_CHANNELS):
    img = tf.py_function(load_img, inp=[img_path], Tout=[tf.float32, tf.int8])
    img[0].set_shape([224,224,NUM_CHANNELS])
    img[1].set_shape([])
    return img