import tensorflow as tf


def PSNR(y_true, y_pred):
    epsilon = 1e-6
    max_val = max(tf.reduce_max(y_true), tf.reduce_max(y_pred), epsilon)
    return tf.image.psnr(y_true, y_pred, max_val)
