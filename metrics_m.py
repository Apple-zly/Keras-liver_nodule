import keras.backend as K
import tensorflow as tf

def dice_lesion(y_true, y_pred):

    y_pred_f = K.softmax(y_pred)
    y_true_f = K.squeeze(y_true, axis=-1)

    y_pred_f = K.argmax(y_pred_f, axis=-1)

    y_pred2_f = K.equal(y_pred_f, 2)
    y_true2_f = K.equal(y_true_f, 2)

    y_pred2_f = tf.cast(y_pred2_f, tf.float32)
    y_true2_f = tf.cast(y_true2_f, tf.float32)

    intersection2 = tf.reduce_sum(y_true2_f * y_pred2_f, axis=[1,2])
    union2 = tf.reduce_sum(y_true2_f, axis=[1,2]) + tf.reduce_sum(y_pred2_f, axis=[1,2])
    dice2 = (2. * intersection2) / (union2 + 0.001)
    dice2 = tf.reduce_mean(dice2)

    if (tf.reduce_sum(y_pred2_f) == 0) and (tf.reduce_sum(y_true2_f) == 0):
        dice2 = 1

    return dice2

def dice_liver(y_true, y_pred):

    y_pred_f = K.softmax(y_pred)
    y_true_f = K.squeeze(y_true, axis=-1)

    y_pred_f = K.argmax(y_pred_f, axis=-1)

    y_pred1_f = K.equal(y_pred_f, 1)
    y_true1_f = K.equal(y_true_f, 1)

    y_pred1_f = tf.cast(y_pred1_f, tf.float32)
    y_true1_f = tf.cast(y_true1_f, tf.float32)

    intersection1 = tf.reduce_sum(y_true1_f * y_pred1_f, axis=[1,2])
    union1 = tf.reduce_sum(y_true1_f, axis=[1,2]) + tf.reduce_sum(y_pred1_f, axis=[1,2])
    dice1 = (2. * intersection1) / (union1 + 0.001)
    dice1 = tf.reduce_mean(dice1)

    if (tf.reduce_sum(y_pred1_f) == 0) and (tf.reduce_sum(y_true1_f) == 0):
        dice1 = 1

    return dice1

def dice_lesion_3d(y_true, y_pred):

    y_pred = y_pred[:,:,:,1:7,:]
    y_true = y_true[:,:,:,1:7,:]

    y_pred_f = K.softmax(y_pred)
    y_true_f = K.squeeze(y_true, axis=-1)

    y_pred_f = K.argmax(y_pred_f, axis=-1)

    y_pred2_f = K.equal(y_pred_f, 2)
    y_true2_f = K.equal(y_true_f, 2)

    y_pred2_f = tf.cast(y_pred2_f, tf.float32)
    y_true2_f = tf.cast(y_true2_f, tf.float32)

    intersection2 = tf.reduce_sum(y_true2_f * y_pred2_f, axis=[1,2,3])
    union2 = tf.reduce_sum(y_true2_f, axis=[1,2,3]) + tf.reduce_sum(y_pred2_f, axis=[1,2,3])
    dice2 = (2. * intersection2) / (union2 + 0.001)
    dice2 = tf.reduce_mean(dice2)

    if (tf.reduce_sum(y_pred2_f) == 0) and (tf.reduce_sum(y_true2_f) == 0):
        dice2 = 1

    return dice2

def dice_liver_3d(y_true, y_pred):

    y_pred = y_pred[:,:,:,1:7,:]
    y_true = y_true[:,:,:,1:7,:]

    y_pred_f = K.softmax(y_pred)
    y_true_f = K.squeeze(y_true, axis=-1)

    y_pred_f = K.argmax(y_pred_f, axis=-1)

    y_pred1_f = K.equal(y_pred_f, 1)
    y_true1_f = K.equal(y_true_f, 1)

    y_pred1_f = tf.cast(y_pred1_f, tf.float32)
    y_true1_f = tf.cast(y_true1_f, tf.float32)

    intersection1 = tf.reduce_sum(y_true1_f * y_pred1_f, axis=[1,2,3])
    union1 = tf.reduce_sum(y_true1_f, axis=[1,2,3]) + tf.reduce_sum(y_pred1_f, axis=[1,2,3])
    dice1 = (2. * intersection1) / (union1 + 0.001)
    dice1 = tf.reduce_mean(dice1)

    if (tf.reduce_sum(y_pred1_f) == 0) and (tf.reduce_sum(y_true1_f) == 0):
        dice1 = 1

    return dice1

def dice_lesion_3d_12(y_true, y_pred):
    
    y_pred = y_pred[:,:,:,1:11,:]
    y_true = y_true[:,:,:,1:11,:]

    y_pred_f = K.softmax(y_pred)
    y_true_f = K.squeeze(y_true, axis=-1)

    y_pred_f = K.argmax(y_pred_f, axis=-1)

    y_pred2_f = K.equal(y_pred_f, 2)
    y_true2_f = K.equal(y_true_f, 2)

    y_pred2_f = tf.cast(y_pred2_f, tf.float32)
    y_true2_f = tf.cast(y_true2_f, tf.float32)

    intersection2 = tf.reduce_sum(y_true2_f * y_pred2_f, axis=[1,2,3])
    union2 = tf.reduce_sum(y_true2_f, axis=[1,2,3]) + tf.reduce_sum(y_pred2_f, axis=[1,2,3])
    dice2 = (2. * intersection2) / (union2 + 0.001)
    dice2 = tf.reduce_mean(dice2)

    if (tf.reduce_sum(y_pred2_f) == 0) and (tf.reduce_sum(y_true2_f) == 0):
        dice2 = 1

    return dice2

def dice_liver_3d_12(y_true, y_pred):

    y_pred = y_pred[:,:,:,1:11,:]
    y_true = y_true[:,:,:,1:11,:]

    y_pred_f = K.softmax(y_pred)
    y_true_f = K.squeeze(y_true, axis=-1)

    y_pred_f = K.argmax(y_pred_f, axis=-1)

    y_pred1_f = K.equal(y_pred_f, 1)
    y_true1_f = K.equal(y_true_f, 1)

    y_pred1_f = tf.cast(y_pred1_f, tf.float32)
    y_true1_f = tf.cast(y_true1_f, tf.float32)

    intersection1 = tf.reduce_sum(y_true1_f * y_pred1_f, axis=[1,2,3])
    union1 = tf.reduce_sum(y_true1_f, axis=[1,2,3]) + tf.reduce_sum(y_pred1_f, axis=[1,2,3])
    dice1 = (2. * intersection1) / (union1 + 0.001)
    dice1 = tf.reduce_mean(dice1)

    if (tf.reduce_sum(y_pred1_f) == 0) and (tf.reduce_sum(y_true1_f) == 0):
        dice1 = 1

    return dice1
