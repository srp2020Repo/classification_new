import tensorflow as tf

@tf.function
def load_img(img_path):

    label = tf.constant(1,tf.int8) if tf.strings.regex_full_match(img_path,".*gctb.*") \
            else tf.constant(0,tf.int8)

    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img)
    img = tf.cast(img, tf.float32) / 255.0
    return (img, label)